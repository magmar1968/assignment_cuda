#ifndef __SCHEDULE__
#define __SCHEDULE__



#define HD __host__ __device__

class Schedule {

	private:
		double * _t;         //array of dates 
		int      _dim;       
		bool     _ascending;
	public:
		HD Schedule(void){};
		HD Schedule(double t_ref, double delta_t, int dim);
		HD Schedule(double* t, int dim);
		HD ~Schedule(void);  

		HD bool Check_order() const;
		HD bool Get_order() const;
		HD double Get_t(int i) const;
		HD void Get_t_vector(double* ptr) const;  //aggiuntiva per avere subito il vettore dei tempi
		HD int Get_dim(void) const;
};

#endif
