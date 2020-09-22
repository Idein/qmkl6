
import sys

from videocore6 import pack_unpack
from videocore6.assembler import assemble, qpu


def ilog2(n):
    '''
    >>> ilog2(1.)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(-1)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(0)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(1)
    0
    >>> ilog2(2)
    1
    >>> ilog2(4)
    2
    >>> ilog2(8)
    3
    '''

    assert isinstance(n, int)
    assert n > 0 and n & (n - 1) == 0
    return n.bit_length() - 1


# Originally written by N. Ohkawa in the py-videocore6 project.

@qpu
def qpu_sgemm_rtn(asm, *, num_qpus, code_offset, align_cond=lambda pos: True):

    # α ⋅ (P × Q) ⋅ (Q × R) + β ⋅ (P × R)
    # α ⋅ (m × k) ⋅ (k × n) + β ⋅ (m × n)
    # α ⋅ (i × k) ⋅ (k × j) + β ⋅ (i × j)

    IDX0_R, IDX0_Q, IDX0_A, IDX0_B, IDX0_C, IDX0_A_CUR, IDX0_B_CUR, \
        IDX0_LDA, IDX0_LDB, IDX0_LDC, IDX0_ALPHA, IDX0_BETA, \
        IDX0_I, IDX0_J, IDX0_K, IDX0_ROTATE_N = range(16)

    # r1 = qpu_num
    if num_qpus == 1:
        mov(r1, 0)
    elif num_qpus == 8:
        tidx(r1)
        shr(r1, r1, 2)
        band(r1, r1, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    eidx(r2).mov(r0, 0)
    for idx in [IDX0_I, IDX0_R, IDX0_Q, IDX0_A, IDX0_B, IDX0_C,
                IDX0_LDA, IDX0_LDB, IDX0_LDC, IDX0_ALPHA, IDX0_BETA]:
        nop(sig=ldunifrf(r5))
        sub(null, r2, idx, cond='pushz')
        mov(r0, r5, cond='ifa')

    # LDA *= 4
    sub(null, r2, IDX0_LDA, cond='pushz')
    shl(r0, r0, ilog2(4), cond='ifa')
    # LDB *= 4
    sub(null, r2, IDX0_LDB, cond='pushz')
    shl(r0, r0, ilog2(4), cond='ifa')
    # LDC *= 4
    sub(null, r2, IDX0_LDC, cond='pushz')
    shl(r0, r0, ilog2(4), cond='ifa')
    # A += 4 * qpu_num
    sub(null, r2, IDX0_A, cond='pushz')
    shl(r3, r1, ilog2(4))
    add(r0, r0, r3, cond='ifa')
    # C += 4 * LDC * qpu_num
    nop()
    rotate(broadcast, r0, -IDX0_LDC)
    sub(null, r2, IDX0_C, cond='pushz').umul24(r3, r5, r1)
    add(r0, r0, r3, cond='ifa')
    # LDC *= num_qpus
    sub(null, r2, IDX0_LDC, cond='pushz')
    shl(r0, r0, ilog2(num_qpus), cond='ifa')

    for i in range(8):
        mov(rf[i], .0).mov(rf[i + 8], .0)

    nop(sig=thrsw)
    nop()
    nop()

    # I = P - 16 * num_qpus - 1
    eidx(r1).mov(r2, 1)
    sub(null, r1, IDX0_I, cond='pushz')
    shl(r2, r2, ilog2(16) + ilog2(num_qpus), cond='ifa')
    sub(r0, r0, r2, cond='ifa')
    sub(r0, r0, 1, cond='ifa')

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as lm:

        # J = R - 16 - 1
        assert IDX0_R == 0
        nop()
        eidx(r1).mov(broadcast, r0)
        sub(null, r1, IDX0_J, cond='pushz')
        add(r5, r5, -16)
        sub(r0, r5, 1, cond='ifa')

        with loop as ln:

            # For I = rest_m - 16 * num_qpus - 1 and cur_m = qpu_num:
            # r4 = rest_m - cur_m = I + 16 * num_qpus + 1 - qpu_num
            if num_qpus == 1:
                mov(r5, 0).mov(r4, 1)
            elif num_qpus == 8:
                tidx(r5).mov(r4, 1)
                shr(r5, r5, 2)
                band(r5, r5, 0b1111)
            shl(r4, r4, ilog2(16) + ilog2(num_qpus))
            sub(r4, r4, r5).rotate(broadcast, r0, -IDX0_I)
            # If rest_m <= cur_m ∴ we have nothing to do, then exit the m-loop.
            add(r4, r4, r5, cond='pushn')
            b(R.exit_m, cond='a0')
            # K = Q - 1
            eidx(r1).rotate(broadcast, r0, -IDX0_Q)
            sub(null, r1, IDX0_K, cond='pushz')
            sub(r0, r5, 1, cond='ifa')

            # A_CUR = A
            # B_CUR = B
            sub(null, r1, IDX0_A_CUR, cond='pushz')
            rotate(broadcast, r0, -IDX0_A)
            mov(r0, r5, cond='ifa')
            sub(null, r1, IDX0_B_CUR, cond='pushz')
            rotate(broadcast, r0, -IDX0_B)
            mov(r0, r5, cond='ifa')

            with loop as lk:

                # Load *(A_CUR + 4 * num_qpus * eidx)
                # and *(B_CUR + 4 * eidx)
                eidx(r1)
                shl(r2, r1, ilog2(4) + ilog2(num_qpus))
                rotate(broadcast, r0, -IDX0_A_CUR)
                shl(r1, r1, ilog2(4)).add(tmua, r5, r2)
                rotate(broadcast, r0, -IDX0_B_CUR)
                add(tmua, r5, r1)

                # A_CUR += 4 * LDA
                # B_CUR += 4 * LDB
                eidx(r1).rotate(broadcast, r0, -IDX0_LDA)
                sub(null, r1, IDX0_A_CUR, cond='pushz')
                add(r0, r0, r5, cond='ifa')
                sub(null, r1, IDX0_B_CUR, cond='pushz')
                rotate(broadcast, r0, -IDX0_LDB)
                add(r0, r0, r5, cond='ifa')

                nop(sig=ldtmu(r1))
                nop(sig=ldtmu(r2))

                # For J = rest_n - 16 - 1:
                # If rest_n - eidx < 1 ∴ J + 16 - eidx < 0, then zero-clear B.
                rotate(broadcast, r0, -IDX0_J)
                eidx(r3).sub(r5, r5, -16)
                sub(null, r5, r3, cond='pushn')
                mov(r2, .0, cond='ifa')

                mov(broadcast, r1).rotate(null, r0, -IDX0_K, cond='pushz')
                for i in range(16):
                    fmul(r3, r5, r2)
                    fadd(rf[i], rf[i], r3).rotate(broadcast, r1, -i - 1)

                lk.b(cond='na0')
                eidx(r1)
                sub(null, r1, IDX0_K, cond='pushz')
                sub(r0, r0, 1, cond='ifa')

            # *(C + 4 * eidx + 4 * LDC * num_qpus * l) for l = 0, 1, ..., 15
            nop()
            eidx(r1).rotate(broadcast, r0, -IDX0_C)
            shl(r1, r1, ilog2(4))
            add(r1, r5, r1)

            eidx(r2).rotate(broadcast, r0, -IDX0_J)
            # For J = rest_n - 16 - 1:
            # r5 = -1 - J = 16 - rest_n ≡ -rest_n (mod 16)
            # r3 = J + 1 = rest_n - 16
            sub(r5, -1, r5).sub(r3, r5, -1)
            # If eidx + rest_n - 16 < 0 ...
            add(null, r2, r3, cond='pushn').mov(r3, r5)
            mov(broadcast, r1).rotate(r1, r1, r5)
            mov(r1, r5, cond='ifa').sub(null, r2, IDX0_ROTATE_N, cond='pushz')
            mov(r0, r3, cond='ifa')

            for i in range(16):
                mov(tmua, r1).rotate(broadcast, r0, -IDX0_ALPHA)
                fmul(r2, rf[i], r5)
                rotate(broadcast, r0, -IDX0_ROTATE_N)
                nop()
                rotate(rf[i], r2, r5)
                nop(sig=ldtmu(r2))
                rotate(broadcast, r0, -IDX0_BETA)
                # If rest_m <= cur_m, then skip the remaining rows of C.
                sub(r4, r4, num_qpus, cond='pushn').fmul(r2, r2, r5)
                b(R.exit_c, cond='a0')
                fadd(tmud, rf[i], r2).mov(rf[i], .0)
                mov(tmua, r1).rotate(broadcast, r0, -IDX0_LDC)
                add(r1, r1, r5)

            L.exit_c

            # B += 4 * 16
            # C += 4 * 16
            eidx(r1).umul24(r2, 8, 8)
            sub(null, r1, IDX0_B, cond='pushz')
            add(r0, r0, r2, cond='ifa')
            sub(null, r1, IDX0_C, cond='pushz')
            add(r0, r0, r2, cond='ifa').rotate(null, r0, -IDX0_J, cond='pushn')
            ln.b(cond='na0')
            eidx(r1)
            sub(null, r1, IDX0_J, cond='pushz')
            add(r0, r0, -16, cond='ifa')

        # A += 4 * 16 * num_qpus
        # B -= 4 * 16 * ⌈ R / 16 ⌉
        # C += 4 * LDC * 16 * num_qpus - 4 * 16 * ⌈ R / 16 ⌉
        eidx(r1).mov(r5, 1)
        sub(null, r1, IDX0_A, cond='pushz')
        shl(r5, r5, ilog2(4) + ilog2(16) + ilog2(num_qpus))
        add(r0, r0, r5, cond='ifa')
        nop()

        rotate(broadcast, r0, -IDX0_LDC)
        sub(null, r1, IDX0_C, cond='pushz')
        shl(r5, r5, ilog2(16))
        add(r0, r0, r5, cond='ifa')
        nop()

        assert IDX0_R == 0
        mov(broadcast, r0)
        add(r5, r5, 15)
        shr(r5, r5, ilog2(16))
        shl(r5, r5, ilog2(4) + ilog2(16))
        sub(r0, r0, r5, cond='ifa')
        sub(null, r1, IDX0_B, cond='pushz')
        sub(r0, r0, r5, cond='ifa')

        mov(r2, 1)
        eidx(r1).rotate(null, r0, -IDX0_I, cond='pushn')
        lm.b(cond='na0')
        sub(null, r1, IDX0_I, cond='pushz')
        shl(r2, r2, ilog2(16) + ilog2(num_qpus))
        sub(r0, r0, r2, cond='ifa')

    L.exit_m

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

    num_qpus, code_offset = map(int, sys.argv[1:])

    for insn in assemble(qpu_sgemm_rtn, num_qpus=num_qpus,
                         code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
