#include <limits.h>
#include "fed.h"

int main()
{
  float b  [100];  /* right hand side                              */
  float x  [100];  /* unknown solution                             */
  float tmp[100]; /* Temporary helper array                        */
  float *omega;    /* Vector of FJ Relaxation Parameters           */
  int   N;         /* Cycle Length                                 */
  int   M;         /* Number of cycles                             */
  int   n, m, i;   /* Loop counters for iterations, cycles, pixels */


  /* Sparse System Matrix A (symmetric, positive definite)         */
  
  /* A_i,i    =  4   (i = 0,...,99)
     A_i,i-10 = -1   (i = 10,...,99)
     A_i,i-1  = -1   (i = 1,...99)
     A_i,i+1  = -1   (i = 0,...,98)
     A_i,i+10 = -1   (i = 0,...,89)
  */

  /* Initialise rhs with uniform random numbers from [0,1]         */
  printf("\n\nright hand side:\n\n");
  for(i = 0; i < 100; i++) 
  {
    b[i] = (float)rand()/(float)RAND_MAX;
    printf("%f\n", b[i]);
  }

  /* Initial vector x^0                                            */
  for(i = 0; i < 100; i++) 
  {
    x[i] = 0.0f;
  }

  
  /* Initialise relaxation parameters for Fast-Jacobi with         */
  /* - cycle length:                      20                       */
  /* - number of cycles M:                5                        */
  /* - omega_max:                         1.0                      */
  M = 5;
  N = fastjac_relax_params(20, 1.0f, 1, &omega);
  
  /* Perform M outer cycles                                        */
  for(m = 0; m < M; m++)
  {
    /* Each cycle performs N iterations with varying parameters    */
    for(n = 0; n < N; n++)
    {      
      /* Compute matrix-vector product tmp = Ax                    */
      for(i = 0; i < 100; i++)
	{
	  tmp[i]  = 4.0f * x[i];
	  tmp[i] += (i>9)  ? -1.0f * x[i-10] : 0.0f;
	  tmp[i] += (i>0)  ? -1.0f * x[i-1]  : 0.0f;
	  tmp[i] += (i<99) ? -1.0f * x[i+1]  : 0.0f;
	  tmp[i] += (i<90) ? -1.0f * x[i+10] : 0.0f;
	}
 
      /* Fast-Jacobi iteration (Note: 1/A_ii = 0.25)               */
      /* x^(n+1) = x^n + omega[n] * D^-1 * (b - Ax^n)              */
      for(i = 0; i < 100; i++)
        x[i] = x[i] + omega[n] * 0.25f * (b[i] - tmp[i]);
    }
  }

  /* Output filtered array                                         */
  printf("\n\nnumerical solution:\n\n");
  for(i = 0; i < 100; i++) 
    printf("%f\n", x[i]);

  /* Free relaxation parameter vector                              */
  free(omega);

  return 0;
}

