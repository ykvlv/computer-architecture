val cap = 4000000;

val prev = 0;
val curr = 1;
val next;

val total = 0;

while (curr < cap) {
    next = prev + curr;
    prev = curr;
    curr = next;

    if (curr % 2 == 0) {
        total = total + curr;
    }
}

val character[12];
val char_ptr = 0;

while (total > 0) {
    character[char_ptr] = total % 10 + 48;
    total = total / 10;
    char_ptr = char_ptr + 1;
}
char_ptr = char_ptr - 1;

val current_number;
while (char_ptr>=0){
    current_number = character[char_ptr];
    put current_number;
    char_ptr = char_ptr - 1;
}
