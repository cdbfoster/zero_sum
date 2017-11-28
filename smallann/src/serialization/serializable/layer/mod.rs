//
// This file is part of smallann.
//
// smallann is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// smallann is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with smallann. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2017 Chris Foster
//

macro_rules! read_layer_types {
    (
        activation_layers: [$(
            $activation_layer:ident<F>,
        )*],
        activation_functions: [$(
            $activation_function:ident,
        )*],
        gradient_descent_layers: [$(
            $gradient_descent_layer:ident<G>,
        )*],
        gradient_descent_algorithms: [$(
            $gradient_descent_algorithm:ident,
        )*],
        other_layers: [$(
            $layer:ident,
        )*],
    ) => {
        identifiable! {
            $($activation_layer<F>,)*
            $($activation_function,)*
            $($gradient_descent_layer<G>,)*
            $($gradient_descent_algorithm,)*
            $($layer,)*
        }
        pub fn read_layer(file: &mut BufReader<File>) -> Result<Box<Layer>> {
            let strings = read_line(file)?;

            if strings.len() < 1 {
                return read_error(file, "Cannot read layer type!");
            }

            read_layer_types!(@expand strings, file, $($activation_layer),*; ($($activation_function),*));
            read_layer_types!(@expand strings, file, $($gradient_descent_layer),*; ($($gradient_descent_algorithm),*));

            $(if strings[0] == $layer::identifier() {
                return $layer::read_from_file(file).map(|l| Box::new(l) as Box<Layer>);
            })*

            return read_error(file, "Unknown layer type!");
        }
    };
    (@expand $strings:ident, $file:ident, $($layer:ident),*; $types:tt) => {
        $(read_layer_types!(@else_if_layer_per_type $strings, $file, $layer, $types );)*
    };
    (@else_if_layer_per_type $strings:ident, $file:ident, $layer:ident, ($($type:ident),*)) => {
        $(if $strings[0] == $layer::<$type>::identifier() {
            return $layer::<$type>::read_from_file($file).map(|l| Box::new(l) as Box<Layer>);
        })*
    };
}

mod activation;
mod composite;
mod convolutional;
mod fully_connected;
mod pass_through;
mod split;
