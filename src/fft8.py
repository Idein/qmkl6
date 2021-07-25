
import sys

from videocore6.assembler import assemble, qpu

from common import ilog2


@qpu
def qpu_fft8(asm, *, num_qpus, is_forward):

    g = globals()
    for i, name in enumerate(['j', '2j', '3j', '4j', '5j', '6j', '7j', 'k', 'l',
                              'm', 'n', '4n', 'x_orig', 'x', 'y_orig', 'y',
                              'buf', 'invsqrt2_neg', 'omega_r', 'omega_c']):
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

    shl(reg_4n, reg_n, ilog2(4))

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

            add(reg_2j, reg_j, reg_j).umul24(reg_7j, reg_j, 7)
            add(reg_3j, reg_j, reg_2j).add(reg_4j, reg_2j, reg_2j)
            add(reg_5j, reg_2j, reg_3j).add(reg_6j, reg_3j, reg_3j)

            with loop as lm:

                # x + 8 * (n / radix * s) = x + n * s, s = 0, 1, ..., 7
                # r3 = x + n * s
                # r4 = x + n * s + n * 4

                mov(tmuau, reg_x).add(r3, reg_x, reg_n)
                add(tmua, reg_x, reg_4n).add(r4, r3, reg_4n)
                mov(tmua, r3).add(r3, r3, reg_n)
                mov(tmua, r4).add(r4, r4, reg_n)
                mov(tmuau, r3).add(r3, r3, reg_n)
                nop(sig=ldtmu(reg_c0r))
                mov(tmua, r4).add(r4, r4, reg_n, sig=ldtmu(reg_c0c))
                nop(sig=ldtmu(reg_c4r))
                mov(tmua, r3) \
                    .mov(broadcast, reg_invsqrt2_neg, sig=ldtmu(reg_c4c))

                # (c0, c1, ..., c7) = butt8(x0, x1, ..., x7)

                fadd(r0, reg_c0r, reg_c4r, sig=ldtmu(reg_c1r))
                fsub(reg_c4r, reg_c0r, reg_c4r) \
                    .mov(tmua, r4, sig=ldtmu(reg_c1c))
                fadd(r0, reg_c0c, reg_c4c).mov(reg_c0r, r0)
                fsub(reg_c4c, reg_c0c, reg_c4c) \
                    .mov(reg_c0c, r0, sig=ldtmu(reg_c5r))

                fadd(r0, reg_c1r, reg_c5r)
                fsub(reg_c5r, reg_c1r, reg_c5r, sig=ldtmu(reg_c5c))
                fadd(r0, reg_c1c, reg_c5c).mov(reg_c1r, r0)
                fsub(reg_c5c, reg_c1c, reg_c5c).mov(reg_c1c, r0)
                if is_forward:
                    fadd(r0, reg_c5r, reg_c5c, sig=ldtmu(reg_c2r))
                    fsub(r1, reg_c5c, reg_c5r) \
                        .fmul(reg_c5r, r0, r5.unpack('abs'), sig=ldtmu(reg_c2c))
                    fmul(reg_c5c, r1, r5.unpack('abs'), sig=ldtmu(reg_c6r))
                else:
                    fsub(r0, reg_c5r, reg_c5c, sig=ldtmu(reg_c2r))
                    fadd(r1, reg_c5r, reg_c5c) \
                        .fmul(reg_c5r, r0, r5.unpack('abs'), sig=ldtmu(reg_c2c))
                    fmul(reg_c5c, r1, r5.unpack('abs'), sig=ldtmu(reg_c6r))

                fadd(r0, reg_c2r, reg_c6r, sig=ldtmu(reg_c6c))
                if is_forward:
                    fsub(r1, reg_c6r, reg_c2r) \
                        .mov(reg_c2r, r0, sig=ldtmu(reg_c3r))
                else:
                    fsub(r1, reg_c2r, reg_c6r) \
                        .mov(reg_c2r, r0, sig=ldtmu(reg_c3r))
                fadd(r0, reg_c2c, reg_c6c, sig=ldtmu(reg_c3c))
                if is_forward:
                    fsub(reg_c6r, reg_c2c, reg_c6c) \
                        .mov(reg_c2c, r0, sig=ldtmu(reg_c7r))
                else:
                    fsub(reg_c6r, reg_c6c, reg_c2c) \
                        .mov(reg_c2c, r0, sig=ldtmu(reg_c7r))

                fadd(r0, reg_c3r, reg_c7r).mov(reg_c6c, r1)
                fsub(reg_c7r, reg_c3r, reg_c7r) \
                    .mov(reg_c3r, r0, sig=ldtmu(reg_c7c))
                fadd(r0, reg_c3c, reg_c7c)
                fsub(reg_c7c, reg_c3c, reg_c7c).mov(reg_c3c, r0)
                if is_forward:
                    fsub(r0, reg_c7c, reg_c7r)
                    fadd(r1, reg_c7c, reg_c7r) \
                        .fmul(reg_c7r, r0, r5.unpack('abs'))
                    mov(r3, reg_y).fmul(reg_c7c, r1, r5)
                else:
                    fadd(r0, reg_c7r, reg_c7c)
                    fsub(r1, reg_c7r, reg_c7c).fmul(reg_c7r, r0, r5)
                    mov(r3, reg_y).fmul(reg_c7c, r1, r5.unpack('abs'))

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

                fadd(r0, reg_c4r, reg_c6r).mov(reg_c3c, r0)
                fadd(r1, reg_c4c, reg_c6c).mov(reg_c3r, r1)
                fsub(reg_c6r, reg_c4r, reg_c6r).mov(reg_c4r, r0)
                fsub(reg_c6c, reg_c4c, reg_c6c).mov(reg_c4c, r1)

                fadd(r0, reg_c5r, reg_c7r)
                fadd(r1, reg_c5c, reg_c7c)
                if is_forward:
                    fsub(r4, reg_c7r, reg_c5r).mov(reg_c5r, r0)
                    fsub(reg_c7r, reg_c5c, reg_c7c).mov(reg_c5c, r1)
                else:
                    fsub(r4, reg_c5r, reg_c7r).mov(reg_c5r, r0)
                    fsub(reg_c7r, reg_c7c, reg_c5c).mov(reg_c5c, r1)

                # eidx < 16 - (m + 1) ∴ m + eidx - 15 < 0
                # r5 = -1 - (rest - 1) = -rest
                eidx(r0).add(r2, reg_m, -15)
                add(null, r0, r2, cond='pushn').sub(r5, -1, reg_m)
                add(reg_m, reg_m, -16, cond='pushn').mov(reg_c7c, r4)
                mov(reg_y, r3, cond='ifb').rotate(reg_y, r3, r5, cond='ifnb')

                # (e0, e1) = butt2(d0, d1)
                # (e2, e3) = butt2(d2, d3)
                # (e4, e5) = butt2(d4, d5)
                # (e6, e7) = butt2(d6, d7)

                fadd(r0, reg_c0r, reg_c1r)
                rotate(tmud, r0, r5)
                fadd(r0, reg_c0c, reg_c1c)
                rotate(tmud, r0, r5)
                mov(tmuau, reg_y)
                nop(sig=ldunifrf(r4))

                fadd(r2, reg_c4r, reg_c5r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c4c, r1).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c3c)
                add(tmua, reg_y, reg_j, sig=ldunifrf(r4))

                fadd(r2, reg_c2r, reg_c3r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c2c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c7c)
                add(tmua, reg_y, reg_2j, sig=ldunifrf(r4))

                fadd(r2, reg_c6r, reg_c7r, sig=ldunifrf(reg_omega_c))
                fadd(r3, reg_c6c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c1c)
                add(tmua, reg_y, reg_3j, sig=ldunifrf(r4))

                fsub(r2, reg_c0r, reg_c1r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c0c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c5c)
                add(tmuau, reg_y, reg_4j)
                nop(sig=ldunifrf(r4))

                fsub(r2, reg_c4r, reg_c5r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c4c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c3c)
                add(tmua, reg_y, reg_5j, sig=ldunifrf(r4))

                fsub(r2, reg_c2r, reg_c3r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c2c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, reg_c7c)
                add(tmua, reg_y, reg_6j, sig=ldunifrf(r4))

                fsub(r2, reg_c6r, reg_c7r, sig=ldunifrf(reg_omega_c))
                fsub(r3, reg_c6c, r0).fmul(r0, r2, r4, sig=rot(r5))
                fmul(r1, r3, reg_omega_c, sig=rot(r5))
                fsub(tmud, r0, r1).fmul(r0, r2, reg_omega_c, sig=rot(r5))
                fmul(r1, r3, r4, sig=rot(r5))
                fadd(tmud, r0, r1).mov(r0, 7)

                lm.b(cond='na0').unif_addr(absolute=False)
                # 7 = ilog2(8 * 16)
                shl(r0, 1, r0).add(r1, reg_l, 1)
                add(reg_x, reg_x, r0).sub(null, r1, reg_k, cond='pushn')
                add(tmua, reg_y, reg_7j).add(reg_y, reg_y, r0)

            ll.b(cond='a0')
            # Restore the original reg_j value.
            shr(reg_j, reg_j, ilog2(8)).mov(reg_l, r1)
            sub(null, reg_j, 2, cond='pushn')
            shr(r0, reg_k, ilog2(8), cond='pushz')

        ljk.b(cond='na0')
        shl(reg_j, reg_j, ilog2(8)).mov(reg_k, r0)
        mov(reg_y_orig, reg_buf, cond='ifb').mov(r0, reg_y_orig)
        mov(reg_y_orig, reg_x_orig, cond='ifnb').mov(reg_x_orig, r0)

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
            twiddle = drv.alloc((n - 1) // 7 * 19, dtype='float32')

            rng = default_rng(0xdeadbeef)
            src[:] = rng.random(n, 'float32') + rng.random(n, 'float32') * 1j
            dst[:] = 0
            tmp[:] = 0

            c = pi / (-4 if is_forward else 4)
            j = 0
            for ilog8k in range(ilog8n - 1, -1, -1):
                k = pow(8, ilog8k)
                for l in range(k):
                    twiddle[j] = unpack('f', pack('I', 0xfafafafa))[0]
                    j += 1
                    twiddle[j] = unpack('f', pack('I', 0xfafafafa))[0]
                    j += 1
                    twiddle[j] = unpack('f', pack('I', 0xfafafafa))[0]
                    j += 1
                    for m in range(1, 5):
                        omega = rect(1, c * l * m / k)
                        twiddle[j], twiddle[j + 1] = omega.real, omega.imag
                        j += 2
                    twiddle[j] = unpack('f', pack('I', 0xfafafafa))[0]
                    j += 1
                    for m in range(5, 8):
                        omega = rect(1, c * l * m / k)
                        twiddle[j], twiddle[j + 1] = omega.real, omega.imag
                        j += 2
                    twiddle[j] = unpack('f', pack('i', -4 * 19))[0]
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
