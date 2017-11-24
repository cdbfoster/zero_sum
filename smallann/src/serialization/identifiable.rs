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

pub trait Identifiable {
    fn identifier() -> String where Self: Sized;
    fn get_identifier(&self) -> String;
}

macro_rules! identifiable {
    ($($type:ident$(<$($T:ident),+>)*,)*) => {
        $(
            identifiable_single!($type$(<$($T),*>)*);
        )*
    };
}

macro_rules! identifiable_single {
    ($type:ident$(<$($T:ident),+>)*) => {
        impl$(<$($T: Identifiable),*>)* Identifiable for $type$(<$($T),*>)* {
            #[allow(unused_mut)]
            fn identifier() -> String {
                let mut identifier = String::from(stringify!($type));
                $(
                    identifier.push('<');
                    $(
                        identifier.push_str(&$T::identifier());
                    )*
                    identifier.push('>');
                )*

                identifier
            }

            fn get_identifier(&self) -> String {
                $type::$(<$($T),*>::)*identifier()
            }
        }
    };
}
