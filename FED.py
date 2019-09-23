"""
Author : Shekoufeh Gorgi Zadeh
Bonn university 
Jun. 2016

"""

from cffi import FFI

class FED(object):
    def __init__(self):
        self.ffi = FFI()
        # List of the functions in the library
        self.ffi.cdef("""
            int fed_tau_by_steps(int n, float tau_max, int reordering, float **tau);
            int fed_tau_by_cycle_time(float t, float tau_max, int reordering, float **tau);
            int fed_tau_by_process_time(float T, int M, float tau_max, int reordering, float **tau);
            float fed_max_cycle_time_by_steps(int n, float tau_max);
            float fed_max_process_time_by_steps(int n, int M, float tau_max);
            int fastjac_relax_params(int n, float omega_max, int reordering, float **omega); 
            """)
        
        # Load library
        self.fedlib = self.ffi.dlopen("fedfjlib//fed//fed.so")
        
    def fed_tau_by_steps(self, n, tau_max, reordering):
        """
        Allocates an array of n time steps and fills it with FED time step sizes,
        such that the maximal stopping time for this cycle is obtained. 
        RETURNS n if everything is ok, or 0 on failure. 
        
        Input:
            n : Desired number of internal steps
            tau_max : Stability limit for explicit (0.5^Dim)   
            reordering : Reordering flag   
        Output:
            tau : Time step widths (allocated inside)
        """
    
        tau = self.ffi.new("float **") # Declare the output
        result = self.fedlib.fed_tau_by_steps(n, tau_max, reordering, tau) # Call the function written in C
        return list(tau[0][0:result])
    
    def fed_tau_by_cycle_time(self, t, tau_max, reordering):
        """
        Allocates an array of the least number of time steps such that a certain  
        stopping time per cycle can be obtained, and fills it with the respective 
        FED time step sizes.                                                                                                                          
        RETURNS number of time steps per cycle, or 0 on failure.   
        
        Input:
            t : Desired cycle stopping time
            tau_max : Stability limit for explicit (0.5^Dim)   
            reordering : Reordering flag   
        Output:
            tau : Time step widths (allocated inside)
        """
    
        tau = self.ffi.new("float **")
        result = self.fedlib.fed_tau_by_cycle_time(t, tau_max, reordering, tau) 
        return list(tau[0][0:result])
    
    def fed_tau_by_process_time(self, T, M, tau_max, reordering):
        """
        Allocates an array of the least number of time steps such that a certain  
        stopping time for the whole process can be obtained, and fills it with       
        the respective FED time step sizes for one cycle.        

        RETURNS number of time steps per cycle, or 0 on failure.   
        
        Input:
            T : Desired process stopping time 
            M : Desired number of cycles 
            tau_max : Stability limit for explicit (0.5^Dim)   
            reordering : Reordering flag   
        Output:
            tau : Time step widths (allocated inside)
        """
    
        tau = self.ffi.new("float **") 
        result = self.fedlib.fed_tau_by_process_time(T, M, tau_max, reordering, tau) 
        return list(tau[0][0:result])
    
    def fed_max_cycle_time_by_steps(self, n, tau_max):
        """
        Computes the maximal cycle time that can be obtained using a certain     
        number of steps. This corresponds to the cycle time that arises from a       
        tau array which has been created using fed_tau_by_steps.                                                   
        RETURNS cycle time t 
        
        Input:
            n : Number of steps per FED cycle
            tau_max : Stability limit for explicit (0.5^Dim)     
        Output:
            t : Cycle time t
        """
    
        return self.fedlib.fed_max_cycle_time_by_steps(n, tau_max) 
    
    def fed_max_process_time_by_steps(self, n, M, tau_max):
        """
        Computes the maximal process time that can be obtained using a certain       
        number of steps. This corresponds to the cycle time that arises from a     
        tau array which has been created using fed_tau_by_steps.                    
                                              
        RETURNS cycle time t 
        
        Input:
            n : Number of steps per FED cycle 
            M : Number of cycles
            tau_max : Stability limit for explicit (0.5^Dim)     
        Output:
            t : Cycle time t
        """
        return self.fedlib.fed_max_process_time_by_steps(n, M, tau_max)
    
    def fastjac_relax_params(self, n, omega_max, reordering):
        """
        Allocates an array of n relaxation parameters and fills it with the FED     
        based parameters for Fast-Jacobi.                      
                                                                                
        RETURNS n if everything is ok, or 0 on failure. 
        
        Input:
            n : Cycle length             
            omega_max : Stability limit for Jacobi over-relax.   
            reordering : Reordering flag   
        Output:
            omega : Relaxation parameters (allocated inside)
        """
    
        omega = self.ffi.new("float **") # Declare the output
        result = self.fedlib.fastjac_relax_params(n, omega_max, reordering, omega) # Call the function written in C
        return list(omega[0][0:result])