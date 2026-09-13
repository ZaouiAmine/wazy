package frontend

import (
	"github.com/samyfodil/wazy/internal/engine/native/ssa"
	"github.com/samyfodil/wazy/internal/platform"
	"github.com/samyfodil/wazy/internal/wasm"
)

// wasm SIMD on a machine with no vector unit.
//
// A v128 is two 64-bit words, and every SIMD operation can be expressed on that
// pair with ordinary integer and floating-point instructions. Where the hardware
// has no vector unit, this lowers the vector opcodes that way instead of into
// ssa.TypeV128 values -- so no v128 value is ever created, the register allocator
// never sees its third register file, and the backend needs no part in it.
//
// This matters because SIMD is not opt-in from the guest's point of view. It is
// part of CoreFeaturesV2, which is wazy's default, so enabling it withheld the
// compiler from *every* module on such a machine -- including the overwhelming
// majority that contain no v128 at all. On riscv64 the vector extension is
// optional and most shipping silicon lacks it, which made the default
// configuration interpret everything.
//
// The expansion lives in the frontend rather than in a later pass because the
// frontend is what owns the places a v128 can appear: the operand stack, block
// parameters, locals, and signatures. Splitting a value into two there is
// bookkeeping; splitting it after the IR is built would mean rewriting phis and
// call sites.

// emulateSIMD reports whether the vector opcodes are lowered to scalar pairs.
//
// A property of the machine, not of a configuration: it is the same answer for
// every module in the process, and a module compiled one way must never be served
// from cache to a runtime expecting the other. Tests override it to exercise the
// scalar paths on a host that has a vector unit, which is the only way this gets
// developed at a reasonable pace.
var emulateSIMD = !platform.SIMDSupported()

// simdEmulated is the ledger of vector opcodes lowered without a vector unit.
//
// Explicit, and consulted before the switch below runs, so an opcode that has no
// scalar lowering yet refuses the module loudly instead of falling through to a
// path that would emit vector instructions the CPU cannot execute. Membership
// here is the coverage number.
var simdEmulated = map[wasm.OpcodeVec]bool{
	wasm.OpcodeVecV128Const:  true,
	wasm.OpcodeVecV128Load:   true,
	wasm.OpcodeVecV128Store:  true,
	wasm.OpcodeVecV128Not:    true,
	wasm.OpcodeVecV128And:    true,
	wasm.OpcodeVecV128Or:     true,
	wasm.OpcodeVecV128Xor:    true,
	wasm.OpcodeVecV128AndNot: true,
	wasm.OpcodeVecI64x2Add:   true,
	wasm.OpcodeVecI64x2Sub:   true,
}

// pushV128 puts a v128 on the operand stack as its two words, low first.
//
// The stack is a flat list of values, so a v128 simply occupies two entries. Every
// count taken from a signature must therefore come from the expanded form rather
// than from the wasm arity, or the two disagree about how deep the stack is.
func (c *Compiler) pushV128(lo, hi ssa.Value) {
	state := c.state()
	state.push(lo)
	state.push(hi)
}

// popV128 takes the two words of a v128 off the operand stack.
func (c *Compiler) popV128() (lo, hi ssa.Value) {
	state := c.state()
	hi = state.pop()
	lo = state.pop()
	return
}

// v128LoadWords reads a v128 as two 64-bit loads from an address the caller has
// already bounds-checked for the full sixteen bytes.
func (c *Compiler) v128LoadWords(addr ssa.Value, disp uint32) (lo, hi ssa.Value) {
	builder := c.ssaBuilder
	loI := builder.AllocateInstruction()
	loI.AsLoad(addr, disp, ssa.TypeI64)
	builder.InsertInstruction(loI)
	hiI := builder.AllocateInstruction()
	hiI.AsLoad(addr, disp+8, ssa.TypeI64)
	builder.InsertInstruction(hiI)
	return loI.Return(), hiI.Return()
}

// v128StoreWords writes a v128 as two 64-bit stores.
func (c *Compiler) v128StoreWords(lo, hi, addr ssa.Value, disp uint32) {
	builder := c.ssaBuilder
	builder.AllocateInstruction().AsStore(ssa.OpcodeStore, lo, addr, disp).Insert(builder)
	builder.AllocateInstruction().AsStore(ssa.OpcodeStore, hi, addr, disp+8).Insert(builder)
}

// v128Binary applies op to both words of two v128 operands, which is the whole
// lowering for any operation that treats the vector as 128 undifferentiated bits
// or as two independent 64-bit lanes.
func (c *Compiler) v128Binary(op func(x, y ssa.Value) ssa.Value) {
	y0, y1 := c.popV128()
	x0, x1 := c.popV128()
	c.pushV128(op(x0, y0), op(x1, y1))
}

// The scalar word operations the lane lowerings are built from. Methods rather
// than closures at each site so the lowering above reads as the operation it is.

func (c *Compiler) scalarBand(x, y ssa.Value) ssa.Value {
	return c.ssaBuilder.AllocateInstruction().AsBand(x, y).Insert(c.ssaBuilder).Return()
}

func (c *Compiler) scalarBor(x, y ssa.Value) ssa.Value {
	i := c.ssaBuilder.AllocateInstruction()
	i.AsBor(x, y) // AsBor and AsBxor return nothing, unlike AsBand
	c.ssaBuilder.InsertInstruction(i)
	return i.Return()
}

func (c *Compiler) scalarBxor(x, y ssa.Value) ssa.Value {
	i := c.ssaBuilder.AllocateInstruction()
	i.AsBxor(x, y)
	c.ssaBuilder.InsertInstruction(i)
	return i.Return()
}

// scalarBnot is xor with all ones: the SSA has no scalar bitwise-not.
func (c *Compiler) scalarBnot(x ssa.Value) ssa.Value {
	ones := c.ssaBuilder.AllocateInstruction().AsIconst64(^uint64(0)).Insert(c.ssaBuilder).Return()
	return c.scalarBxor(x, ones)
}

// scalarBandnot is x & ^y, matching v128.andnot's operand order.
func (c *Compiler) scalarBandnot(x, y ssa.Value) ssa.Value {
	return c.scalarBand(x, c.scalarBnot(y))
}

func (c *Compiler) scalarIadd(x, y ssa.Value) ssa.Value {
	return c.ssaBuilder.AllocateInstruction().AsIadd(x, y).Insert(c.ssaBuilder).Return()
}

func (c *Compiler) scalarIsub(x, y ssa.Value) ssa.Value {
	return c.ssaBuilder.AllocateInstruction().AsIsub(x, y).Insert(c.ssaBuilder).Return()
}

// withSIMDEmulation forces the mode and returns a function restoring it, so a
// test can pin which lowering it is describing instead of inheriting the host's
// CPU -- and so the scalar paths can be exercised on a machine that has a vector
// unit, which is the only way they get developed at a reasonable pace.
func withSIMDEmulation(on bool) func() {
	prev := emulateSIMD
	emulateSIMD = on
	return func() { emulateSIMD = prev }
}

// SetEmulateSIMD forces the mode for the whole process and returns a function
// restoring it. Exported for the spec suites, which run the scalar lowering
// against the specification's own SIMD assertions on whatever host is to hand:
// that is the only gate strong enough for code that reimplements every vector
// operation, and waiting for riscv64 hardware to run it would be no gate at all.
//
// Process-wide, and it changes what the compiler emits, so it is for tests. A
// compilation cache is keyed on the module and its features, not on this, so do
// not share one across a change of mode.
func SetEmulateSIMD(on bool) (restore func()) { return withSIMDEmulation(on) }
