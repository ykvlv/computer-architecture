val divisors[20];
val n = 2;
val current_number;
val div_ptr;

while (n < 21) {
    current_number = n;
    div_ptr = 0;
    while ( (div_ptr < 20) & divisors[div_ptr] ){
        if (current_number % divisors[div_ptr] == 0) {
            current_number = current_number / divisors[div_ptr];
        }
        div_ptr = div_ptr + 1;
    }
    if ( current_number > 1 ){
        divisors[div_ptr] = current_number;
    }
    n = n + 1;
}

val result = 1;
div_ptr = 0;
while (divisors[div_ptr]){
    result = result * divisors[div_ptr];
    div_ptr = div_ptr + 1;
}

val character[12];
val char_ptr = 0;
while (result > 0) {
    character[char_ptr] = result % 10 + 48;
    result = result / 10;
    char_ptr = char_ptr + 1;
}
char_ptr = char_ptr - 1;

while (char_ptr>=0){
    current_number = character[char_ptr];
    put current_number;
    char_ptr = char_ptr - 1;
}


