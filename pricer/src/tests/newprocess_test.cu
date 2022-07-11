#include "../lib/support_lib/myRandom/myRandom.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/combined.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/tausworth.cuh"
#include "../lib/support_lib/myRandom/myRandom_gnr/linCongruential.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal_multivariante.cuh"
#include "../lib/path_gen_lib/process_eq_imp/process_eq_lognormal.cuh"
#include "../lib/equity_lib/schedule_lib/schedule.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve.cuh"
#include "../lib/equity_lib/yield_curve_lib/yield_curve_flat.cuh"
#include "../lib/support_lib/parse_lib/parse_lib.cuh"
#include "../lib/contract_option_lib/contract_eq_option_vanilla/contract_eq_option_vanilla.cuh"
#include "../lib/support_lib/statistic_lib/statistic_lib.cuh"
#include "../lib/support_lib/timer_lib/myTimer.cuh"
//#include "../lib/support_lib/myDouble_lib/myudouble.cuh"

#define NEQ 1

struct Input_data
{
	char contract_type;
	double strike_price;
	double delta_t;
	double vol;
	char isin_code[12];
	char name[30];
	char currency[20];
	double div_yield;
	double yc;
	double start_prices[NEQ];
	double seeds[4];
};

struct Dimensions
{
	int BLOCKS;
	int TPB;
};

struct Output_data
{
	double sum;
	double sq_sum;
};

__global__ void kernel(Input_data*, Output_data*);
D void simulate_device(Input_data*, Output_data*);
H void simulate_host(Input_data*, Output_data*, Dimensions*);
HD void simulate_generic(Input_data*, Output_data*, size_t);

__global__ void kernel(Input_data* input_data, Output_data* output_data)
{
	simulate_device(input_data, output_data);
}

D void simulate_device(Input_data* input_data, Output_data* output_data)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	simulate_generic(input_data, output_data, index);
}

H void simulate_host(Input_data* input_data, Output_data* output_data, Dimensions* dimensions)
{
    for(size_t index = 0; index < dimensions->BLOCKS * dimensions->TPB; index++)
	{
		simulate_generic(input_data, output_data, index);
	}
}

HD void simulate_generic(Input_data* input_data, Output_data* output_data, size_t index)
{
	output_data[index].sum = input_data[index].strike_price;
	output_data[index].sq_sum = input_data[index].start_prices[0];
}

void Gen_dimensions(Dimensions* dim, int a, int b)
{
	dim->BLOCKS = 128 * pow(2, a);
	dim->TPB = 256 * pow(2, b);
}


int main(int argc, char** argv)
{
	cudaError_t cudaStatus;
	srand(time(NULL));
	Dimensions* dim = new Dimensions;
	Timer _timer;
	for (int t = 0; t < 100; t++)
	{
		//printf("Progresso: %d di 100", t);
		
		for (size_t a = 1; a < 6; a++)
		{
			for (size_t b = 1; b < 3; b++)
			{
				Gen_dimensions(dim, a, b);
				int blocchi = dim->BLOCKS;
				int tpb = dim->TPB;

				Input_data* host_in = new Input_data[blocchi * tpb];
				Output_data* host_out = new Output_data[blocchi * tpb];

				uint seed_aus[4];
				for (size_t i = 0; i < 4; i++)
				{
					seed_aus[i] = rnd::genSeed(true);
				}
				rnd::GenCombined gnr_aus(seed_aus[0], seed_aus[1], seed_aus[2], seed_aus[3]);
				for (size_t i = 0; i < blocchi * tpb; i++)
				{
					for (size_t j = 0; j < 4; j++)
					{
						host_in[i].seeds[j] = gnr_aus.genUniformInt();
						while (host_in[i].seeds[j] <= 128)
						{
							host_in[i].seeds[j] = gnr_aus.genUniformInt();
						}
					}
					host_in[i].contract_type = 'C';
					host_in[i].strike_price = 100;
					host_in[i].delta_t = 0.2;
					host_in[i].vol = 0.0001;
					strcpy(host_in[i].isin_code, "123456789012");
					strcpy(host_in[i].name, "prova");
					strcpy(host_in[i].currency, "euro");
					host_in[i].div_yield = 0;
					host_in[i].yc = 0.05;
					host_in[i].start_prices[0] = 100;
				}

				prcr::Device dev;
				dev.CPU = false;
				dev.GPU = false;

				if (prcr::cmdOptionExists(argv, argv + argc, "-gpu"))
					dev.GPU = true;
				if (prcr::cmdOptionExists(argv, argv + argc, "-cpu"))
					dev.CPU = true;

				bool error_bool;
				error_bool = true;
				if (dev.CPU == true)
				{
					simulate_host(host_in, host_out, dim);
				}

				if (dev.GPU == true)
				{
					Input_data* dev_in;// = new Input_data[blocchi * tpb]; 
					Output_data* dev_out;//= new Output_data[blocchi * tpb];
					//Dimensions* dev_dim = new Dimensions;

					cudaStatus = cudaMalloc((void**)&dev_in, blocchi * tpb * sizeof(Input_data));
					if (cudaStatus != cudaSuccess) 
					{
						fprintf(stderr, "cudaMalloc1 failed!\n"); 
						printf("Errore con blocchi : %d, tpb: %d \n", blocchi, tpb);
						error_bool = false;
					}

					cudaStatus = cudaMalloc((void**)&dev_out, blocchi * tpb * sizeof(Output_data));
					if (cudaStatus != cudaSuccess)
					{ 
						fprintf(stderr, "cudaMalloc2 failed!\n");
						printf("Errore con blocchi : %d, tpb: %d \n", blocchi, tpb);
						error_bool = false;
					}

					/*cudaStatus = cudaMalloc((void**)&dev_dim, sizeof(Dimensions));
					if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc3 failed!\n"); }*/

					cudaStatus = cudaMemcpy(dev_in, host_in, blocchi * tpb * sizeof(Input_data), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess)
					{ 
						fprintf(stderr, "cudaMemcpy1 failed! %s\n", cudaGetErrorString(cudaStatus));
						printf("Errore con blocchi : %d, tpb: %d \n", blocchi, tpb);
						error_bool = false;
					}

					/*cudaStatus = cudaMemcpy(dev_dim, dim, sizeof(Dimensions), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy2 failed!\n"); }
					fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));*/

					kernel << < blocchi, tpb >> > (dev_in, dev_out);
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) 
					{
						fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); 
						printf("Errore con blocchi : %d, tpb: %d \n", blocchi, tpb);
						error_bool = false;
					}

					cudaFree(dev_in);

					cudaStatus = cudaMemcpy(host_out, dev_out, blocchi * tpb * sizeof(Output_data), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess)
					{
						fprintf(stderr, "cudaMemcpy backwards failed! %s\n", cudaGetErrorString(cudaStatus));
						printf("Errore con blocchi : %d, tpb: %d \n", blocchi, tpb);
						error_bool = false;
					}

					cudaFree(dev_out);




				}

				for (int i = 0; i < blocchi * tpb; i++)
				{
					if (error_bool)
					{
						// std::cout << host_out[].sum <<"   "  << host_out[i].sq_sum << std::endl;
						if (host_out[i].sum - 100 != 0) { printf("errore in sum: numero blocchi %d, numero tpb %d,  indice %d\n", blocchi, tpb, i); }
						if (host_out[i].sq_sum != 100) { printf("errore in sq_sum: numero blocchi %d, numero tpb %d, indice %d\n", blocchi, tpb, i); }
					}
				}
			}
		}
	}
	delete(dim);
	_timer.Stop();
	return 0;
}
