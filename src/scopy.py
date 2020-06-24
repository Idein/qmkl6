
import sys

from videocore6.assembler import assemble, qpu


@qpu
def qpu_scopy(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: pos % 512 == 259):

    g = globals()
    for i, v in enumerate(['length', 'src', 'src_inc', 'src_stride', 'dst',
                           'dst_inc', 'dst_stride', 'qpu_num']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_src))
    nop(sig=ldunifrf(reg_src_inc))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_dst_inc))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * (thread_num + 16 * qpu_num) * inc
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    umul24(r1, r0, reg_src_inc)
    add(reg_src, reg_src, r1).umul24(r1, r0, reg_dst_inc)
    add(reg_dst, reg_dst, r1)

    # stride = 4 * 16 * num_qpus * inc
    mov(r0, 1)
    shl(r0, r0, 6 + num_qpus_shift)
    umul24(reg_src_stride, r0, reg_src_inc)
    umul24(reg_dst_stride, r0, reg_dst_inc)

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, 7 + num_qpus_shift + unroll_shift)

    # This single thread switch and two nops just before the loop are really
    # important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(8):
            mov(tmua, reg_src).add(reg_src, reg_src, reg_src_stride)

        for j in range(unroll - 1):
            for i in range(8):
                nop(sig=ldtmu(r0))
                mov(tmua, reg_src).add(reg_src, reg_src, reg_src_stride)
                mov(tmud, r0)
                mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_dst_stride)

        for i in range(6):
            nop(sig=ldtmu(r0))
            mov(tmud, r0)
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_dst_stride)

        nop(sig=ldtmu(r0))
        mov(tmud, r0).sub(reg_length, reg_length, 1, cond='pushz')
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_dst_stride)

        l.b(cond='na0')
        nop(sig=ldtmu(r0))                                        # delay slot
        mov(tmud, r0)                                             # delay slot
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_dst_stride)  # delay slot

    # This synchronization is needed between the last TMU operation and the
    # program end with the thread switch just before the loop above.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def main():

    num_qpus, unroll_shift, code_offset = map(int, sys.argv[1:])

    for insn in assemble(qpu_scopy, num_qpus=num_qpus,
                         unroll_shift=unroll_shift, code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
