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
		HD virtual ~Schedule(void);

		HD double* Get_t(int i);
		HD int Get_dim(void);
};

#endif // !__SCHEDULE__