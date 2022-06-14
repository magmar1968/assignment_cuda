#ifndef __SCHEDULE__
#define __SCHEDULE__



#define H __host__
#define D __device__
#define HD __host__ __device__

class Schedule {

	private:
		double* _t;
		int _dim;
		bool _ascending;
	public:
		HD Schedule(void);
		HD Schedule(double t_ref, double delta_t, int dim);
		HD Schedule(double* t_init, int dim);
		//HD virtual ~Schedule(void);  //? virtual

		HD bool Check_order() const;
		HD bool Get_order() const;
		HD double Get_t(int i) const;
		HD void Get_t_vector(double* ptr) const;  //aggiuntiva per avere subito il vettore dei tempi
		HD int Get_dim(void) const;
};

#endif