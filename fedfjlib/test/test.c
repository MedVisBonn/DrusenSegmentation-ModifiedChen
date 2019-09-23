#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int _fed_is_prime_internal(int number)
{
  int i;
  int s = (int)sqrtf((float)number + 1.0f);

  for (i = 2; i <= s; ++i)
    if (!(number % i))
      return 0;

  return 1;
}

int main()
{
	int num_to_check = 20;
	printf("%d is %d\n",num_to_check,_fed_is_prime_internal(num_to_check));
}
