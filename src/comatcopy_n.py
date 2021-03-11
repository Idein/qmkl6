
import sys

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
    assert n > 0
    assert n & (n - 1) == 0

    return n.bit_length() - 1


@qpu
def qpu_comatcopy_n(asm, *, num_qpus, unroll, code_offset,
                    align_cond=lambda pos: True):

    assert unroll in [1, 2, 4]

    g = globals()
    for i, v in enumerate(['rows', 'cols', 'alpha_r', 'alpha_i', 'a', 'lda',
                           'b', 'ldb', 'qpu_num', 'a_i', 'b_i', 'inc', 'rest_i',
                           'rest_j']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_rows))
    nop(sig=ldunifrf(reg_cols))
    nop(sig=ldunifrf(reg_alpha_r))
    nop(sig=ldunifrf(reg_alpha_i))
    nop(sig=ldunifrf(reg_a))
    nop(sig=ldunifrf(reg_lda))
    nop(sig=ldunifrf(reg_b))
    nop(sig=ldunifrf(reg_ldb))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # a += 8 * lda * qpu_num + 8 * eidx
    # b += 8 * ldb * qpu_num + 8 * eidx
    eidx(r1).umul24(r0, reg_lda, reg_qpu_num)
    shl(r1, r1, ilog2(8)).add(reg_a, reg_a, r0)
    add(reg_a, reg_a, r1).add(reg_a_i, reg_a, r1)
    umul24(r0, reg_ldb, reg_qpu_num)
    add(reg_b, reg_b, r0)
    add(reg_b, reg_b, r1).add(reg_b_i, reg_b, r1)

    # lda *= num_qpus
    # ldb *= num_qpus
    umul24(reg_lda, reg_lda, num_qpus)
    umul24(reg_ldb, reg_ldb, num_qpus)

    # inc = 8 * 16 = 128
    mov(r0, 1)
    shl(reg_inc, r0, ilog2(128))

    # rest_i = rows - qpu_num - 1
    sub(r0, reg_rows, reg_qpu_num)
    sub(reg_rest_i, r0, 1)

    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as li:

        shr(r0, reg_cols, ilog2(16) + ilog2(unroll), cond='pushz')
        b(R.rest_j, cond='a0')
        nop()
        nop()
        sub(reg_rest_j, r0, 2, cond='pushn')

        for i in range(unroll - 2):
            mov(tmua, reg_a).mov(r5, 4)
            add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)

        mov(tmua, reg_a).mov(r5, 4)
        b(R.skip_j_unroll, cond='a0')
        add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)
        if unroll >= 2:
            mov(tmua, reg_a).mov(r5, 4)
            add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)
        else:
            nop()
            nop()

        with loop as lj:

            sub(reg_rest_j, reg_rest_j, 1, cond='pushn')

            for i in range(unroll - 1):
                nop(sig=ldtmu(r0))
                fmul(r2, r0, reg_alpha_r, sig=ldtmu(r1))
                fmul(r3, r1, reg_alpha_i)
                fsub(tmud, r2, r3).fmul(r2, r0, reg_alpha_i)
                mov(tmua, reg_b).fmul(r3, r1, reg_alpha_r)
                mov(tmua, reg_a)
                fadd(tmud, r2, r3).mov(r5, 4)
                add(tmua, reg_b, r5).add(reg_b, reg_b, reg_inc)
                add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)

            nop(sig=ldtmu(r0))
            fmul(r2, r0, reg_alpha_r, sig=ldtmu(r1))
            fmul(r3, r1, reg_alpha_i)
            fsub(tmud, r2, r3).fmul(r2, r0, reg_alpha_i)
            mov(tmua, reg_b).fmul(r3, r1, reg_alpha_r)
            mov(tmua, reg_a)
            lj.b(cond='na0')
            fadd(tmud, r2, r3).mov(r5, 4)
            add(tmua, reg_b, r5).add(reg_b, reg_b, reg_inc)
            add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)

        L.skip_j_unroll

        for i in range(unroll):
            nop(sig=ldtmu(r0))
            fmul(r2, r0, reg_alpha_r, sig=ldtmu(r1))
            fmul(r3, r1, reg_alpha_i)
            fsub(tmud, r2, r3).fmul(r2, r0, reg_alpha_i)
            mov(tmua, reg_b).fmul(r3, r1, reg_alpha_r)
            fadd(tmud, r2, r3).mov(r5, 4)
            add(tmua, reg_b, r5).add(reg_b, reg_b, reg_inc)

        L.rest_j

        mov(r0, 1)
        shl(r0, r0, ilog2(16) + ilog2(unroll))
        sub(r0, r0, 1)
        band(r0, reg_cols, r0, cond='pushz')
        b(R.exit_j, cond='a0')
        sub(r0, r0, 1)
        add(r0, r0, -16, cond='pushn').mov(r4, r0)
        nop()
        with loop as lj0:
            lj0.b(cond='na0')
            mov(tmua, reg_a).mov(r5, 4)
            add(tmua, reg_a, r5).add(reg_a, reg_a, reg_inc)
            add(r0, r0, -16, cond='pushn')
        with loop as lj1:
            sub(r5, -1, r4)
            nop(sig=ldtmu(r0))
            fmul(r2, r0, reg_alpha_r, sig=ldtmu(r1))
            fmul(r3, r1, reg_alpha_i)
            fsub(r0, r2, r3).fmul(r2, r0, reg_alpha_i)
            fmul(r3, r1, reg_alpha_r)
            fadd(r1, r2, r3).rotate(r0, r0, r5)
            mov(tmud, r0).mov(r3, reg_b)
            rotate(r1, r1, r5)
            rotate(r3, r3, r5)

            mov(broadcast, reg_b)
            eidx(r0).add(r2, r4, -16)
            add(null, r0, r2, cond='pushn')
            mov(r3, r5, cond='ifa')
            add(r4, r4, -16, cond='pushn')
            lj1.b(cond='na0')
            mov(tmua, r3)
            mov(tmud, r1).mov(r5, 4)
            add(tmua, r3, r5).add(reg_b, reg_b, reg_inc)

        L.exit_j

        sub(reg_rest_i, reg_rest_i, num_qpus, cond='pushn')
        li.b(cond='na0')
        add(reg_a_i, reg_a_i, reg_lda)
        add(reg_b_i, reg_b_i, reg_ldb)
        mov(reg_a, reg_a_i).mov(reg_b, reg_b_i)

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

    num_qpus, unroll, code_offset = map(int, sys.argv[1:])

    for insn in assemble(qpu_comatcopy_n, num_qpus=num_qpus, unroll=unroll,
                         code_offset=code_offset):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
