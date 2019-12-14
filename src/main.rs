use julia::{JuliaContext, JuliaData};

#[macro_use]
extern crate gramit;
use gramit::{Vec2, Vec4};

use structopt::StructOpt;

use palette::named;
use palette::Srgb;

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, StructOpt)]
#[structopt(name = "julia", about = "A generator of Julia sets")]
/// Create images of Julia sets.
///
/// julia generates Julia sets for complex polynomials of the form:
///
///     f(x) = x^n + c
///
/// where `n` is an integer and `c` is a complex number `c_r + c_i * i`.
struct JuliaArgs {
    /// The exponent n.
    #[structopt(short = "n", long = "exponent", default_value = "2")]
    n: u32,

    /// The real part of the complex number `c`.
    #[structopt(short = "r", long = "real-part", default_value = "0.0")]
    cr: f32,

    /// The imaginary part of the complex number `c`.
    #[structopt(short = "i", long = "imaginary-part", default_value = "0.0")]
    ci: f32,

    /// The number of iterations to compute.
    #[structopt(short = "m", long = "iters", default_value = "500")]
    iters: u32,

    /// The pixel width of the output image.
    #[structopt(short, long, default_value = "1024")]
    size: u32,

    /// The color gradient. Consists of at least two and at most three comma-separated color names,
    /// and an optional comma-separated value between 0 and 1. Valid color names are those from the
    /// CSS3 specification.
    #[structopt(short, long, parse(try_from_str = parse_gradient),
        default_value = "black,white")]
    colors: ([Vec4; 3], f32),

    /// The name of the output image.
    #[structopt(short = "o", long = "out-file", default_value = "julia.png")]
    file: PathBuf,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct ParseGradientError;

impl Display for ParseGradientError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "failed to parse color gradient")
    }
}

impl Error for ParseGradientError {}

fn parse_gradient(s: &str) -> Result<([Vec4; 3], f32), ParseGradientError> {
    let mut components = s.split(',').map(str::trim);

    fn must_be_color(s: Option<&str>) -> Result<Srgb<f32>, ParseGradientError> {
        let c = s.ok_or(ParseGradientError)?;
        let c = named::from_str(c).ok_or(ParseGradientError)?;
        Ok(Srgb::from_format(c))
    }

    fn to_vec4(c: Srgb<f32>) -> Vec4 {
        let (r, g, b) = c.into_components();
        vec4!(r, g, b, 1.0)
    }

    let c1 = must_be_color(components.next())?;
    let c2 = must_be_color(components.next())?;

    let (c3, midpt) = match components.next() {
        // No third component, flat gradient between two colors
        None => (Srgb::<f32>::new(1.0, 1.0, 1.0), 0.999),
        Some(s) => match named::from_str(s) {
            // Third component isn't a color, truncated gradient between two colors
            None => (
                c2.clone(),
                f32::from_str(s).map_err(|_| ParseGradientError)?,
            ),
            // Third component is a color, see if there's a fourth
            Some(c) => {
                let c = Srgb::<f32>::from_format(c);
                match components.next() {
                    // No midpoint specified, use default
                    None => (c, 0.25),
                    // Midpoint specified
                    Some(s) => (c, f32::from_str(s).map_err(|_| ParseGradientError)?),
                }
            }
        },
    };

    if components.next().is_some() {
        Err(ParseGradientError)
    } else {
        Ok(([to_vec4(c1), to_vec4(c2), to_vec4(c3)], midpt))
    }
}

fn main() {
    let args = JuliaArgs::from_args();
    println!("{:#?}", args);

    let context = JuliaContext::new().expect("failed to create JuliaContext");

    let data = JuliaData {
        color: args.colors.0,
        color_midpoint: args.colors.1,
        n: args.n,
        c: vec2!(args.cr, args.ci),

        iters: args.iters,

        dimensions: args.size,
    };

    context.export(&data, &args.file);
}
