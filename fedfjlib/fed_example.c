#include <limits.h>
#include "fed.h"

int main()
{
  float u  [102]; /* Signal ([1..100] with 1-px bounds)            */
  float tmp[102]; /* Temporary helper array                        */
  float *tau;     /* Vector of FED time step sizes                 */
  int   N;        /* Number of steps                               */
  int   M;        /* Number of cycles                              */
  int   n, m, i;  /* Loop counters over steps, cycles, pixels      */

  /* Inititalise signal with uniform random numbers from [0,1]     */
  printf("\n\ninitial signal:\n\n");
  for(i = 1; i <= 100; i++) 
  {
    u[i] = (float)rand()/(float)RAND_MAX;
    printf("%f\n", u[i]);
  }
  
  /* Initialise step sizes for process with                        */
  /* - overall stopping time T:           500                      */
  /* - number of cycles M:                5                        */
  /* - stability limit for 1-D diffusion: 0.5                      */
  M = 5;
  N = fed_tau_by_process_time(500.0f, M, 0.5f, 1, &tau);
  
  /* Perform M outer cycles                                        */
  for(m = 0; m < M; m++)
  {
    /* Each cycle performs N steps with varying step size          */
    for(n = 0; n < N; n++)
    {
      /* Reflecting boundary conditions                            */
      u[0]   = u[1];
      u[101] = u[100];
      
      /* One explict step                                          */
      for(i = 1; i <= 100; i++)
        tmp[i] = u[i] + tau[n] * (u[i-1] - 2.0f * u[i] + u[i+1]);
        
      /* Copy back for next iteration                              */
      for(i = 1; i <= 100; i++)
        u[i] = tmp[i];
    }
  }

  /* Output filtered array                                         */
  printf("\n\nFiltered:\n\n");
  for(i = 1; i <= 100; i++) 
    printf("%f\n", u[i]);

  /* Free time step vector                                         */
  free(tau);

  return 0;
}

