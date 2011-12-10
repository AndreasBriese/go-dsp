/*
 * Copyright (c) 2011 Matt Jibson <matt.jibson@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package fft

import (
	"errors"
	"math"
)

// radix-2 factors
var factors = map[int][]complex128{}

// bluestein factors
var n2_factors = map[int][]complex128{}
var n2_inv_factors = map[int][]complex128{}
var n2_conj_factors = map[int][]complex128{}

func FFT_real(x []float64) []complex128 {
	return FFT(ToComplex(x))
}

func IFFT_real(x []float64) []complex128 {
	return IFFT(ToComplex(x))
}

// ToComplex takes a []float64 and converts it to []complex128
func ToComplex(x []float64) []complex128 {
	y := make([]complex128, len(x))
	for n, v := range x {
		y[n] = complex(v, 0)
	}
	return y
}

// IFFT calculates the inverse fast Fourier transform of x.
func IFFT(x []complex128) []complex128 {
	lx := len(x)
	r := make([]complex128, lx)

	// Reverse inputs, which is calculated with modulo N, hence x[0] as an outlier
	r[0] = x[0]
	for i := 1; i < lx; i++ {
		r[i] = x[lx-i]
	}

	r = FFT(r)

	N := complex(float64(lx), 0)
	for n, _ := range r {
		r[n] /= N
	}
	return r
}

func Convolve(x, y []complex128) ([]complex128, error) {
	if len(x) != len(y) {
		return []complex128{}, errors.New("fft: input arrays are not of equal length")
	}

	fft_x := FFT(x)
	fft_y := FFT(y)

	r := make([]complex128, len(x))
	for i := 0; i < len(r); i++ {
		r[i] = fft_x[i] * fft_y[i]
	}

	return IFFT(r), nil
}

// FFT calculates the fast Fourier transform of x.
func FFT(x []complex128) []complex128 {
	l := len(x)

	// TODO: non-hack handling length <= 1 cases.
	if l <= 1 {
		r := make([]complex128, l)
		copy(r,x)
		return r
	}
	if IsPowerOf2(l) {
		return Radix2FFT(x)
	}
	return BluesteinFFT(x)
}

func Radix2FFT(x []complex128) []complex128 {
	lx := len(x)

	for i := 4; i <= lx; i <<= 1 {
		if _, present := factors[i]; !present {
			factors[i] = make([]complex128, i)
			for n := 0; n < i; n++ {
				sin, cos := math.Sincos(-2 * float64(n) * math.Pi / float64(i))
				factors[i][n] = complex(cos, sin)
			}
		}
	}

	lx_2 := lx / 2
	r := make([]complex128, lx) // result
	t := make([]complex128, lx)	// temp
	copy(r, x)

	// split into even and odd parts for each stage
	for b := lx; b > 1; b >>= 1 {
		i := 0
		b_2 := b / 2
		for blk := 0; blk < lx/b; blk++ {
			for n := 0; n < b_2; n++ {
				bn := b*blk + n
				t[bn] = r[i]
				i++
				t[bn+b_2] = r[i]
				i++
			}
		}
		copy(r, t)
	}

	for stage := 2; stage <= lx; stage <<= 1 {
		if stage == 2 {
			// 2-point transforms
			for n := 0; n < lx_2; n++ {
				t[n*2] = r[n*2] + r[n*2+1]
				t[n*2+1] = r[n*2] - r[n*2+1]
			}
		} else {
			// >2-point transforms
			blocks := lx / stage
			s_2 := stage / 2

			for n := 0; n < blocks; n++ {
				nb := n * stage
				for j := 0; j < s_2; j++ {
					w_n := r[j+nb+s_2] * factors[stage][j]
					t[j+nb] = r[j+nb] + w_n
					t[j+nb+s_2] = r[j+nb] - w_n
				}
			}
		}

		copy(r, t)
	}
	return r
}

func BluesteinFFT(x []complex128) []complex128 {
	lx := len(x)
	a := ZeroPad(x, NextPowerOf2(lx*2-1))
	la := len(a)

	if _, present := n2_factors[lx]; !present {
		n2_factors[lx] = make([]complex128, lx)
		n2_inv_factors[lx] = make([]complex128, lx)

		for i := 0; i < lx; i++ {
			sin, cos := math.Sincos(math.Pi / float64(lx) * float64(math.Pow(i,2)))
			n2_factors[lx][i] = complex(cos, sin)
			n2_inv_factors[lx][i] = complex(cos, -sin)
		}
	}

	for i, v := range x {
		a[i] = v * n2_inv_factors[lx][i]
	}

	b := make([]complex128, la)
	for i := 0; i < lx; i++ {
		b[i] = n2_factors[lx][i]
		if i != 0 {
			b[la-i] = n2_factors[lx][i]
		}
	}

	// NOTE: The only error Convolve returns is if a and
	// b are not the same length.  Here, b is initialized
	// to len(a), so it's okay to ignore the error.
	r, _ := Convolve(a, b)

	for i := 0; i < lx; i++ {
		r[i] *= n2_inv_factors[lx][i]
	}

	return r[:lx]
}

// IsPowerOf2 returns whether or not x is a power of 2.
func IsPowerOf2(x int) bool {
	return x&(x-1) == 0
}

// NextPowerOf2 returns the smallest power
// of 2 greater than or equal to x.
func NextPowerOf2(x int) int {
	if IsPowerOf2(x) {
		return x
	}
	return int(math.Pow(2, math.Ceil(math.Log2(float64(x)))))
}

// ZeroPad appends zero values to x to the specified length.
func ZeroPad(x []complex128, length int) []complex128 {
	if len(x) == length {
		return x
	}
	r := make([]complex128, length)
	copy(r, x)
	return r
}

// ZeroPad2 calls ZeroPad with the length set to
// the next power of 2 greater than or equal to x.
func ZeroPad2(x []complex128) []complex128 {
	return ZeroPad(x, NextPowerOf2(len(x)))
}

