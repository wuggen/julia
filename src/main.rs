use julia::interface::JuliaInterface;
use julia::{ImgDimensions, JuliaContext, JuliaData};

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
    #[structopt(short, long, default_value = "800")]
    width: u32,

    /// The pixel height of the output image.
    #[structopt(short, long, default_value = "800")]
    height: u32,

    /// The color gradient. Consists of at least two and at most three comma-separated color names,
    /// and an optional comma-separated value between 0 and 1. Valid color names are those from the
    /// CSS3 specification.
    #[structopt(short, long, parse(try_from_str = parse_gradient),
        default_value = "black,white")]
    colors: ([Vec4; 3], f32),

    /// The complex number at the center of the image, given as two comma-separated decimal values.
    #[structopt(short = "O", long, parse(try_from_str = parse_vec2),
        default_value = "0.0,0.0")]
    center: Vec2,

    /// The extent on the complex plane of the largest image dimension.
    #[structopt(short, long, default_value = "3.6")]
    extent: f32,

    /// The name of the output image.
    #[structopt(short = "o", long = "output")]
    file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct ParseGradientError;

impl Display for ParseGradientError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "failed to parse color gradient")
    }
}

impl Error for ParseGradientError {}

impl JuliaArgs {
    fn filename(&self) -> PathBuf {
        match &self.file {
            Some(path) => path.clone(),
            None => {
                fn to_hex(c: Vec4) -> String {
                    let c = Srgb::new(c[0], c[1], c[2]);
                    let c = Srgb::<u8>::from_format(c);
                    format!("{:x}{:x}{:x}", c.red, c.green, c.blue)
                }

                PathBuf::from(format!(
                    "x{}_{}_{}i_m{}_c{}-{}_e{}_c{}-{}-{}-{}_{}x{}.png",
                    self.n,
                    self.cr,
                    self.ci,
                    self.iters,
                    self.center[0],
                    self.center[1],
                    self.extent,
                    to_hex(self.colors.0[0]),
                    to_hex(self.colors.0[1]),
                    to_hex(self.colors.0[2]),
                    self.colors.1,
                    self.width,
                    self.height,
                ))
            }
        }
    }
}

fn parse_hexcode(s: &str) -> Option<Srgb<u8>> {
    eprintln!("Parsing '{}' as hex code", s);
    let mut channels = Vec::new();
    let mut current = 0u8;
    for (i, c) in s.chars().enumerate() {
        eprintln!("Char {}: {}", i, c);
        eprintln!("Parsed channels: {:?}", channels);
        eprintln!("Current channel: {:x}", current);
        if channels.len() >= 3 || (i == 0 && c != '#') {
            eprintln!("Aborting!");
            return None;
        } else {
            if i == 0 {
                continue;
            }

            current = current << 4;
            current += c.to_digit(16)? as u8;
            if i % 2 == 0 {
                channels.push(current);
                current = 0;
            }

            eprintln!(
                "After char {}, current = {}, channels = {:?}",
                i, current, channels
            );
        }
    }

    if channels.len() != 3 {
        None
    } else {
        Some(Srgb::new(channels[0], channels[1], channels[2]))
    }
}

fn parse_gradient(s: &str) -> Result<([Vec4; 3], f32), ParseGradientError> {
    let mut components = s.split(',').map(str::trim);

    fn must_be_color(s: Option<&str>) -> Result<Srgb<f32>, ParseGradientError> {
        let c = s.ok_or(ParseGradientError)?;
        let c = named::from_str(c)
            .or_else(|| parse_hexcode(c))
            .ok_or(ParseGradientError)?;
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
        None => (Srgb::<f32>::new(1.0, 1.0, 1.0), 1.0),
        Some(s) => match named::from_str(s).or_else(|| parse_hexcode(s)) {
            // Third component isn't a color, truncated gradient between two colors
            None => (c2, f32::from_str(s).map_err(|_| ParseGradientError)?),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct ParseVecError;

impl Display for ParseVecError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "failed to parse vec2")
    }
}

impl Error for ParseVecError {}

fn parse_vec2(s: &str) -> Result<Vec2, ParseVecError> {
    let mut components = s.split(',').map(str::trim);

    let x = components.next().ok_or(ParseVecError)?;
    let x = f32::from_str(x).map_err(|_| ParseVecError)?;

    let y = components.next().ok_or(ParseVecError)?;
    let y = f32::from_str(y).map_err(|_| ParseVecError)?;

    if components.next().is_some() {
        Err(ParseVecError)
    } else {
        Ok(vec2!(x, y))
    }
}

fn main() {
    let args = JuliaArgs::from_args();
    println!("{:#?}", args);
    println!("Computed filename: {:?}", args.filename());

    let context = JuliaContext::new().expect("failed to create JuliaContext");

    let dims = ImgDimensions {
        width: args.width,
        height: args.height,
    };

    let aspect = (args.width as f32) / (args.height as f32);
    let (extent_x, extent_y) = if args.width < args.height {
        (args.extent * aspect, args.extent)
    } else {
        (args.extent, args.extent / aspect)
    };

    let data = JuliaData {
        color: args.colors.0,
        color_midpoint: args.colors.1,
        n: args.n,
        c: vec2!(args.cr, args.ci),

        iters: args.iters,

        center: args.center,
        extents: vec2!(extent_x, extent_y),
    };

    context.export(dims, &data, &args.filename());

    //let mut interface = JuliaInterface::new(&context, None).expect("failed to create JuliaInterface");
    //interface.run(&context);
}
