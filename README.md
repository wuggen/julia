A GPU-accelerated Julia set visualizer, using Vulkan!

Interactively generates Julia set visualizations for complex polynomials of the
form `f(x) = x^n + c`.

Written in Rust (and some GLSL). You will need Vulkan development headers in
order to compile, but there shouldn't be any other non-Rust dependencies. Build
with

```
cargo build
```

and run with

```
cargo run
```

## Command-line options

No command-line options are required, and all of their parameters can be changed
interactively. The default settings bring up the Julia set for `f(x) = x^2`,
i.e. a unit circle, visualized with a black background and a white foreground.

- `--exponent <integer>` or `-n <integer>` -- The `n` in `x^n + c`. Default is
  2.
- `--real-part <float>` or `-r <float>` -- The real part of the constant `c` in
  `x^n + c`. Default is 0.
- `--imaginary-part <float>` or `-i <float>` -- The imaginary part of the
  constant `c` in `x^n + c`. Default is 0.
- `--center <complex>` or `-O <complex>` -- The complex number at the center of
  the image. `<complex>` is a comma-separated list of exactly two floating point
  values. Default is `0.0,0.0`.
- `--extents <float>` or `-e <float>` -- The extent on the complex plane covered
  by the largest image dimension. (This is essentially zoom; smaller numbers
  zoom in closer.) Default it 3.6.
- `--iters <integer>` or `-m <integer>` -- The number of iterations per point in
  the visualization generation. Default is 100.
- `--width <integer>` or `-w <integer>` -- The width in pixels of the
  interactive image. Default is 800.
- `--height <integer>` or `-h <integer>` -- The height in pixels of the
  interactive image. Default is 800.
- `--colors <colors>` or `-c <colors>` -- The colors for the visualization.
  `<colors>` is a comma-separated list of at least two and at most three color
  specifications, which may be either named colors from the CSS3 specification,
  or hex codes of the form `#aabbcc`. The third color, if absent, defaults to
  white. Default is `black,white`.
- `--midpts <midpoints>` or `-g <midpoints>` -- The gradient points for the
  visualization. `<midpoints>` is a comma-separated list of at least one and at
  most three values between 0.0 and 1.0, and are interpretted as follows:

  * If only one value is given, it is the gradient point of the second color.
    The first and third colors default to gradient points 0.0 and 1.0
    respectively.
  * If two values are given, they are the gradient points of the first and
    second colors respectively. The third color defaults to gradient point 1.0.
  * If three values are given, they are the gradient points of the three colors
    respectively.

  Default is `0.5`.

## Interactive interface

When julia starts up, it will display a window containing the visualization. If
it was not started from the command line, it _should_ also open a command line
window, since that's where all of the non-visual components of the interface
live. Controls are (almost) entirely keyboard-based. It's a very janky
interface.

To pan the viewport around the complex plane, simply click and drag in the
viewing window. You can also use WASD to move the viewport, (hold Shift to make
smaller steps). To zoom in and out, either scroll in the viewing window with the
mouse wheel, or use the keyboard Plus and Minus keys.

To re-center the image, press C. To reset the zoom, press Z.

To change the polynomial `c`, use the arrow keys. Up and down will change the
imaginary part, while left and right will change the real part. Holding Ctrl
will make `c` take bigger steps, Alt will make it take smaller steps, and Shift
will make it take even smaller steps; Alt+Shift will make it take the smallest
steps of all.

To change the exponent `n`, use PageUp and PageDown.

To change the iteration count of the visualization, use the left and right
square bracket keys; `[` will decrease the iterations, and `]` will increase
them. More iterations yield higher detail in the generated image, but might also
make your computer rebel against you for torture, so tread lightly.

To export a PNG of the current visualization, press E. The file name will be
auto-generated based on the current parameters of the visualization. To increase
and decrease the export resolution width, press I and K respectively. To do the
same for the export resolution height, press O and L.

To modify the visualization gradient, notice that a particular color code and
gradient point are enclosed in [square brackets] in the text interface. These
are the current _active_ color and gradient point. To set the active color, use
the number keys 1, 2, and 3. To set the active gradient point, use 4, 5, and 6.

To change the active color's hue, use the R and F keys. To change its
saturation, use T and G. To change its value, use Y and H.

To modify the active gradient point, use U and J.

To exit, either close the viewing window, or press Q or Esc.
