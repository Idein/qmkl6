
import sys

from videocore6.assembler import assemble, qpu

from common import ilog2


@qpu
def qpu_fft8(asm, *, num_qpus, is_forward):

    g = globals()
    for i, name in enumerate(['j', 'k', 'l', 'm', 'n', 'x_orig', 'x', 'y_orig',
                              'y', 'buf', 'invsqrt2_neg', 'omega_r',
                              'omega_c']):
        g[f'reg_{name}'] = rf[i]

    base = 48
    assert i < base
    for i in range(8):
        g[f'reg_c{i}r'] = rf[base + 2 * i + 0]
        g[f'reg_c{i}c'] = rf[base + 2 * i + 1]

    nop(sig=ldunifrf(reg_n))
    nop(sig=ldunifrf(reg_x_orig))
    nop(sig=ldunifrf(reg_y_orig))
    nop(sig=ldunifrf(reg_buf))
    nop(sig=ldunifrf(reg_invsqrt2_neg))

    nop(sig=thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(null, r0, 0b1111, cond='pushz')
    if num_qpus == 8:
        b(R.exit, cond='na0')
        nop()
        nop()
        nop()
    else:
        raise Exception('num_qpus must be 8')

    b(R.set_unif, cond='always').unif_addr(absolute=True)
    nop()
    nop()
    nop()
    L.set_unif

    shr(reg_k, reg_n, ilog2(8))
    mov(reg_j, 1)
    with loop as ljk:

        mov(reg_l, 0)
        with loop as ll:

            # m = j - 1
            # x = x_orig + (j * l + eidx) * 8
            # y = y_orig + (8 * j * l + eidx) * 8
            # Increase reg_j by eight times in the m loop.
            sub(reg_m, reg_j, 1).mov(r3, reg_j)
            eidx(r0).umul24(r1, reg_j, reg_l)
            shl(r2, r1, ilog2(8)).add(r1, r0, r1)
            shl(r1, r1, ilog2(8)).add(r2, r2, r0)
            shl(r2, r2, ilog2(8)).add(reg_x, reg_x_orig, r1)
            shl(reg_j, r3, ilog2(8)).add(reg_y, reg_y_orig, r2)

            with loop as lm:

                mov(tmua, reg_x).mov(r0, reg_x)
                add(tmua, r0, 4).add(r0, r0, reg_n)
                mov(tmua, r0)
                add(tmua, r0, 4).add(r0, r0, reg_n)
                mov(tmua, r0).mov(r2, 4)
                add(tmua, r0, 4).add(r0, r0, reg_n)
                mov(tmua, r0).mov(r3, reg_y)
                add(tmua, r0, 4).add(r0, r0, reg_n)
                mov(tmua, r0, sig=ldtmu(reg_c0r))
                add(tmua, r0, 4).add(r0, r0, reg_n, sig=ldtmu(reg_c0c))
                mov(tmua, r0, sig=ldtmu(reg_c1r))
                add(tmua, r0, 4).add(r0, r0, reg_n, sig=ldtmu(reg_c1c))
                mov(tmua, r0, sig=ldtmu(reg_c2r))
                add(tmua, r0, 4).add(r0, r0, reg_n, sig=ldtmu(reg_c2c))
                mov(tmua, r0, sig=ldtmu(reg_c3r))
                add(tmua, r0, 4, sig=ldtmu(reg_c3c))
                mov(broadcast, reg_invsqrt2_neg, sig=ldtmu(reg_c4r))

                # (c0, c1, ..., c7) = butt8(x0, x1, ..., x7)

                fadd(r0, reg_c0r, reg_c4r, sig=ldtmu(reg_c4c))
                fadd(r1, reg_c0c, reg_c4c)
                fsub(reg_c4r, reg_c0r, reg_c4r).mov(reg_c0r, r0)
                fsub(reg_c4c, reg_c0c, reg_c4c) \
                    .mov(reg_c0c, r1, sig=ldtmu(reg_c5r))

                fadd(r0, reg_c1r, reg_c5r, sig=ldtmu(reg_c5c))
                fadd(r1, reg_c1c, reg_c5c)
                fsub(reg_c5r, reg_c1r, reg_c5r).mov(reg_c1r, r0)
                fsub(reg_c5c, reg_c1c, reg_c5c).mov(reg_c1c, r1)
                if is_forward:
                    fadd(r0, reg_c5r, reg_c5c)
                    fsub(r1, reg_c5c, reg_c5r) \
                        .fmul(reg_c5r, r0, r5.unpack('abs'))
                    fmul(reg_c5c, r1, r5.unpack('abs'), sig=ldtmu(reg_c6r))
                else:
                    fsub(r0, reg_c5r, reg_c5c)
                    fadd(r1, reg_c5r, reg_c5c) \
                        .fmul(reg_c5r, r0, r5.unpack('abs'))
                    fmul(reg_c5c, r1, r5.unpack('abs'), sig=ldtmu(reg_c6r))

                fadd(r0, reg_c2r, reg_c6r, sig=ldtmu(reg_c6c))
                fadd(r1, reg_c2c, reg_c6c)
                if is_forward:
                    fsub(r0, reg_c6r, reg_c2r).mov(reg_c2r, r0)
                    fsub(r1, reg_c2c, reg_c6c).mov(reg_c2c, r1)
                else:
                    fsub(r0, reg_c2r, reg_c6r).mov(reg_c2r, r0)
                    fsub(r1, reg_c6c, reg_c2c).mov(reg_c2c, r1)
                mov(reg_c6r, r1).mov(reg_c6c, r0, sig=ldtmu(reg_c7r))

                fadd(r0, reg_c3r, reg_c7r, sig=ldtmu(reg_c7c))
                fadd(r1, reg_c3c, reg_c7c)
                fsub(reg_c7r, reg_c3r, reg_c7r).mov(reg_c3r, r0)
                fsub(reg_c7c, reg_c3c, reg_c7c).mov(reg_c3c, r1)
                if is_forward:
                    fsub(r0, reg_c7c, reg_c7r)
                    fadd(r1, reg_c7c, reg_c7r) \
                        .fmul(reg_c7r, r0, r5.unpack('abs'))
                    fmul(reg_c7c, r1, r5)
                else:
                    fadd(r0, reg_c7r, reg_c7c)
                    fsub(r1, reg_c7r, reg_c7c).fmul(reg_c7r, r0, r5)
                    fmul(reg_c7c, r1, r5.unpack('abs'))

                # (d0, d1, d2, d3) = butt4(c0, c1, c2, c3)
                # (d4, d5, d6, d7) = butt4(c4, c5, c6, c7)

                fadd(r0, reg_c0r, reg_c2r)
                fadd(r1, reg_c0c, reg_c2c)
                fsub(reg_c2r, reg_c0r, reg_c2r).mov(reg_c0r, r0)
                fsub(reg_c2c, reg_c0c, reg_c2c).mov(reg_c0c, r1)

                fadd(r0, reg_c1r, reg_c3r)
                fadd(r1, reg_c1c, reg_c3c)
                if is_forward:
                    fsub(r0, reg_c3r, reg_c1r).mov(reg_c1r, r0)
                    fsub(r1, reg_c1c, reg_c3c).mov(reg_c1c, r1)
                else:
                    fsub(r0, reg_c1r, reg_c3r).mov(reg_c1r, r0)
                    fsub(r1, reg_c3c, reg_c1c).mov(reg_c1c, r1)
                mov(reg_c3r, r1).mov(reg_c3c, r0)

                fadd(r0, reg_c4r, reg_c6r)
                fadd(r1, reg_c4c, reg_c6c)
                fsub(reg_c6r, reg_c4r, reg_c6r).mov(reg_c4r, r0)
                fsub(reg_c6c, reg_c4c, reg_c6c).mov(reg_c4c, r1)

                fadd(r0, reg_c5r, reg_c7r)
                fadd(r1, reg_c5c, reg_c7c)
                if is_forward:
                    fsub(r0, reg_c7r, reg_c5r).mov(reg_c5r, r0)
                    fsub(r1, reg_c5c, reg_c7c).mov(reg_c5c, r1)
                else:
                    fsub(r0, reg_c5r, reg_c7r).mov(reg_c5r, r0)
                    fsub(r1, reg_c7c, reg_c5c).mov(reg_c5c, r1)
                mov(reg_c7r, r1).mov(reg_c7c, r0)

                # eidx < 16 - (m + 1) ∴ m + eidx - 15 < 0
                # r5 = -1 - (rest - 1) = -rest
                eidx(r0).add(r1, reg_m, -15)
                add(null, r0, r1, cond='pushn').sub(r5, -1, reg_m)
                add(reg_m, reg_m, -16, cond='pushn')
                mov(reg_y, r3, cond='ifb').rotate(reg_y, r3, r5, cond='ifnb')

                # (e0, e1) = butt2(d0, d1)
                # (e2, e3) = butt2(d2, d3)
                # (e4, e5) = butt2(d4, d5)
                # (e6, e7) = butt2(d6, d7)

                fadd(r0, reg_c0r, reg_c1r)
                mov(r1, reg_y).rotate(tmud, r0, r5)
                fadd(r0, reg_c0c, reg_c1c).mov(tmua, r1)
                add(reg_y, r1, reg_j).rotate(tmud, r0, r5)
                add(tmua, r1, r2).mov(r0, reg_c5c, sig=ldunifrf(r4))

                fadd(r2, reg_c4r, reg_c5r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c4c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c3c, sig=ldunifrf(r4))
                fadd(r2, reg_c2r, reg_c3r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c2c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c7c, sig=ldunifrf(r4))
                fadd(r2, reg_c6r, reg_c7r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c6c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c1c, sig=ldunifrf(r4))
                fsub(r2, reg_c0r, reg_c1r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c0c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c5c, sig=ldunifrf(r4))
                fsub(r2, reg_c4r, reg_c5r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c4c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c3c, sig=ldunifrf(r4))
                fsub(r2, reg_c2r, reg_c3r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c2c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 4)
                add(tmua, reg_y, r0).add(reg_y, reg_y, reg_j)

                mov(r0, reg_c7c, sig=ldunifrf(r4))
                fsub(r2, reg_c6r, reg_c7r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c6c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                mov(tmua, reg_y).fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 7)

                lm.b(cond='na0').unif_addr(absolute=False)
                # 7 = ilog2(8 * 16)
                shl(r0, 1, r0).umul24(r1, reg_j, r0)
                add(tmua, reg_y, 4).sub(reg_y, reg_y, r1)
                add(reg_x, reg_x, r0).add(reg_y, reg_y, r0)

            add(reg_l, reg_l, 1)
            sub(null, reg_l, reg_k, cond='pushn')
            ll.b(cond='a0')
            # Restore the original reg_j value.
            shr(reg_j, reg_j, ilog2(8))
            nop()
            nop()

        shr(reg_k, reg_k, ilog2(8), cond='pushz').mov(r5, ilog2(8))
        ljk.b(cond='na0')
        shl(reg_j, reg_j, r5).sub(null, reg_j, 2, cond='pushn')
        mov(reg_y_orig, reg_buf, cond='ifa').mov(r0, reg_y_orig)
        mov(reg_y_orig, reg_x_orig, cond='ifna').mov(reg_x_orig, r0)

    L.exit

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


def benchmark():

    from cmath import rect
    from math import pi, sqrt
    from struct import pack, unpack
    from time import monotonic

    from numpy.fft import fft, ifft
    from numpy.random import default_rng

    from videocore6.driver import Driver

    def run(n, is_forward, drv, unif, src, dst, tmp, twiddle, num_qpus,
            code_offset=0):

        ref = fft(src) if is_forward else ifft(src, norm='forward')

        code = drv.program(qpu_fft8, num_qpus=num_qpus, is_forward=is_forward)

        unif[0] = n
        unif[1] = src.addresses()[0]
        unif[2] = tmp.addresses()[0]
        unif[3] = dst.addresses()[0]
        if is_swapped:
            unif[2], unif[3] = unif[3], unif[2]
        unif[4] = unpack('I', pack('f', -sqrt(2) / 2))[0]
        unif[5] = twiddle.addresses()[0]

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        err_abs = abs(dst - ref)
        err_rel = abs(err_abs / ref)

        print(', '.join([f'n = 2^{ilog2(n)}', f'{end - start} seconds',
                         f'{n / (end - start) * 1e-6} Melem/s',
                         f'err_abs = [{min(err_abs)}, {max(err_abs)}]',
                         f'err_rel = [{min(err_rel)}, {max(err_rel)}]']))

    num_qpus = 8
    is_forward = True

    for ilog8n in range(1, 8):
        n = pow(8, ilog8n)
        is_swapped = ilog8n % 2 != 0

        with Driver(data_area_size=8 * n * 5 + 4096) as drv:

            unif = drv.alloc(6, dtype='uint32')
            src = drv.alloc(n, dtype='csingle')
            dst = drv.alloc(n, dtype='csingle')
            tmp = drv.alloc(n, dtype='csingle')
            twiddle = drv.alloc((n - 1) // 7 * 15, dtype='float32')

            rng = default_rng(0xdeadbeef)
            src[:] = rng.random(n, 'float32') + rng.random(n, 'float32') * 1j
            dst[:] = 0
            tmp[:] = 0

            c = pi / (-4 if is_forward else 4)
            j = 0
            for ilog8k in range(ilog8n - 1, -1, -1):
                k = pow(8, ilog8k)
                for l in range(k):
                    for m in range(1, 8):
                        omega = rect(1, c * l * m / k)
                        twiddle[j], twiddle[j + 1] = omega.real, omega.imag
                        j += 2
                    twiddle[j] = unpack('f', pack('i', 4 * -15))[0]
                    j += 1
            assert j == len(twiddle)

            run(n, is_forward, drv, unif, src, dst, tmp, twiddle, num_qpus)


def main():

    if len(sys.argv) == 1:
        benchmark()
        return

    num_qpus, is_forward = map(int, sys.argv[1:])

    for insn in assemble(qpu_fft8, num_qpus=num_qpus, is_forward=is_forward):
        print(f'UINT64_C({insn:#018x}),')


if __name__ == '__main__':

    main()
