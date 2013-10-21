#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "new_types/new_types_definitions.hpp"

#ifdef ONLY_INSTANTIATION
 #define EXTERN extern
#else
 #define EXTERN
#endif

namespace nissa
{
  //nomenclature: 
  //-glb is relative to the global grid
  //-loc to the local one
  EXTERN int glb_size[4],glb_vol,glb_spat_vol,glb_volh;
  EXTERN int loc_size[4],loc_vol,loc_spat_vol,loc_volh;
  EXTERN int bulk_vol,non_bw_surf_vol,non_fw_surf_vol;
  EXTERN int surf_vol,bw_surf_vol,fw_surf_vol;
  EXTERN int vsurf_vol,vsurf_volh;
  EXTERN int vdir_bord_vol,vdir_bord_volh;
  EXTERN double glb_vol2,loc_vol2;
  //-lx is lexicografic
  EXTERN coords *glb_coord_of_loclx;
  EXTERN coords *loc_coord_of_loclx;
  EXTERN int *glblx_of_loclx;
  EXTERN int *glblx_of_bordlx;
  EXTERN int *loclx_of_bordlx;
  EXTERN int *surflx_of_bordlx;
  EXTERN int *Wsklx_of_loclx;
  EXTERN int *loclx_of_Wsklx;
  //EXTERN int *Wsklx_hopping_matrix_output_pointer;
  //EXTERN int *Wsklx_hopping_matrix_final_output;
  EXTERN int *glblx_of_edgelx;
  EXTERN int *loclx_of_bulklx;
  EXTERN int *loclx_of_surflx;
  EXTERN int *loclx_of_non_bw_surflx;
  EXTERN int *loclx_of_non_fw_surflx;
  EXTERN int *loclx_of_bw_surflx;
  EXTERN int *loclx_of_fw_surflx;
  EXTERN int lx_geom_inited;
#ifdef USE_VNODES
  EXTERN int vir_geom_inited;
#endif
  EXTERN int Wsklx_order_inited;
  //-eo is even-odd
  EXTERN int *loclx_parity;
  EXTERN int *loceo_of_loclx;
  EXTERN int *loclx_of_loceo[2];
  EXTERN int *surfeo_of_bordeo[2];
  EXTERN coords *loceo_neighup[2];
  EXTERN coords *loceo_neighdw[2];
  EXTERN int eo_geom_inited;
  EXTERN int use_eo_geom;
  
  //neighbours of local volume + borders
  EXTERN coords *loclx_neighdw,*loclx_neighup;
  EXTERN coords *loclx_neigh[2];
  
#ifdef USE_MPI
  //basic mpi types
  EXTERN MPI_Datatype MPI_FLOAT_128;
  EXTERN MPI_Datatype MPI_SU3;
  EXTERN MPI_Datatype MPI_QUAD_SU3;
  EXTERN MPI_Datatype MPI_COLOR;
  EXTERN MPI_Datatype MPI_SPIN;
  EXTERN MPI_Datatype MPI_SPINSPIN;
  EXTERN MPI_Datatype MPI_SPINCOLOR;
  EXTERN MPI_Datatype MPI_SPINCOLOR_128;
  EXTERN MPI_Datatype MPI_REDSPINCOLOR;
  
  //float 128 summ
  EXTERN MPI_Op MPI_FLOAT_128_SUM;
#endif
  
//timings
  EXTERN double tot_time;
#ifdef BENCH
 #ifdef ONLY_INSTANTIATION
   EXTERN double tot_comm_time;
   EXTERN double cgm_inv_over_time,cg_inv_over_time;
   EXTERN int ncgm_inv,ncg_inv;
   EXTERN double portable_stD_app_time;
   EXTERN int portable_stD_napp;
   EXTERN int nsto;
   EXTERN double sto_time;
   EXTERN int nsto_remap;
   EXTERN double sto_remap_time;
   EXTERN int nglu_comp;
   EXTERN double glu_comp_time;
  #ifdef BGQ
    EXTERN double bgq_stdD_app_time;
    EXTERN int bgq_stdD_napp;
  #endif
 #else
   EXTERN double tot_comm_time=0;
   EXTERN double cgm_inv_over_time=0,cg_inv_over_time=0;
   EXTERN int ncgm_inv=0,ncg_inv=0;
   EXTERN double portable_stD_app_time=0;
   EXTERN int portable_stD_napp=0;
   EXTERN int nsto=0;
   EXTERN double sto_time=0;
   EXTERN int nsto_remap=0;
   EXTERN double sto_remap_time=0;
   EXTERN int nglu_comp=0;
   EXTERN double glu_comp_time=0;
  #ifdef BGQ
    EXTERN double bgq_stdD_app_time=0;
    EXTERN int bgq_stdD_napp=0;
  #endif
 #endif
#endif

  //nissa_config parameters
  EXTERN int verb_call;
  EXTERN int verbosity_lv;
  EXTERN int warn_if_not_disallocated;
  EXTERN int use_async_communications;
  EXTERN int warn_if_not_communicated;
  EXTERN coords fix_nranks;
  EXTERN int use_128_bit_precision;
  EXTERN int vnode_paral_dir;
  
  //size of the border and edges
  EXTERN int bord_vol,bord_volh;
  EXTERN int edge_vol,edge_volh;
  //size along various dir
  EXTERN int bord_dir_vol[4],bord_offset[4];
  EXTERN int bord_offset_eo[2][8]; //eo, 8 dirs
  EXTERN int edge_dir_vol[6],edge_offset[6];
  EXTERN int edge_numb[4][4]
#ifndef ONLY_INSTANTIATION
  ={{-1,0,1,2},{0,-1,3,4},{1,3,-1,5},{2,4,5,-1}}
#endif
    ;
#ifdef USE_VNODES
  EXTERN int vnode_lx_offset,vnode_eo_offset;
  EXTERN int vbord_vol,vbord_volh;
  EXTERN int vir_loc_size[4];
#endif
#ifdef USE_MPI
  EXTERN int start_lx_bord_send_up[4],start_lx_bord_rece_up[4];
  EXTERN int start_lx_bord_send_dw[4],start_lx_bord_rece_dw[4];
  EXTERN int start_eo_bord_send_up[4],start_eo_bord_rece_up[4];
  EXTERN int start_eo_bord_send_dw[4],start_eo_bord_rece_dw[4];
  EXTERN MPI_Datatype MPI_EO_QUAD_SU3_BORDS_SEND_TXY[4],MPI_EO_QUAD_SU3_BORDS_RECE[4];
  EXTERN MPI_Datatype MPI_EV_QUAD_SU3_BORDS_SEND_Z[2],MPI_OD_QUAD_SU3_BORDS_SEND_Z[2];
  EXTERN MPI_Datatype MPI_EO_COLOR_BORDS_SEND_TXY[4],MPI_EO_COLOR_BORDS_RECE[4];
  EXTERN MPI_Datatype MPI_EV_COLOR_BORDS_SEND_Z[2],MPI_OD_COLOR_BORDS_SEND_Z[2];
  EXTERN MPI_Datatype MPI_EO_SPIN_BORDS_SEND_TXY[4],MPI_EO_SPIN_BORDS_RECE[4];
  EXTERN MPI_Datatype MPI_EV_SPIN_BORDS_SEND_Z[2],MPI_OD_SPIN_BORDS_SEND_Z[2];
  EXTERN MPI_Datatype MPI_EO_SPINCOLOR_BORDS_SEND_TXY[4],MPI_EO_SPINCOLOR_BORDS_RECE[4];
  EXTERN MPI_Datatype MPI_EV_SPINCOLOR_BORDS_SEND_Z[2],MPI_OD_SPINCOLOR_BORDS_SEND_Z[2];
  EXTERN MPI_Datatype MPI_EO_SPINCOLOR_128_BORDS_SEND_TXY[4],MPI_EO_SPINCOLOR_128_BORDS_RECE[4];
  EXTERN MPI_Datatype MPI_EV_SPINCOLOR_128_BORDS_SEND_Z[2],MPI_OD_SPINCOLOR_128_BORDS_SEND_Z[2];
  
  EXTERN MPI_Datatype MPI_LX_SU3_EDGES_SEND[6],MPI_LX_SU3_EDGES_RECE[6];
  EXTERN MPI_Datatype MPI_LX_QUAD_SU3_EDGES_SEND[6],MPI_LX_QUAD_SU3_EDGES_RECE[6];
  EXTERN MPI_Datatype MPI_EO_QUAD_SU3_EDGES_SEND[96],MPI_EO_QUAD_SU3_EDGES_RECE[6];
  
  //volume, plan and line communicator
  EXTERN MPI_Comm cart_comm;
  EXTERN MPI_Comm plan_comm[4];
  EXTERN MPI_Comm line_comm[4];
#endif
  //ranks
  EXTERN int rank,nranks,cart_rank;
  EXTERN coords rank_coord;
  EXTERN coords rank_neigh[2],rank_neighdw,rank_neighup;
  EXTERN coords plan_rank,line_rank,line_coord_rank;
  EXTERN coords nrank_dir;
  EXTERN int grid_inited;
  EXTERN int nparal_dir;
  EXTERN coords paral_dir;
  
/////////////////////////////////////////////// threads //////////////////////////////////////////
#ifdef USE_THREADS

 #ifdef THREAD_DEBUG
   EXTERN int glb_barr_line;
   EXTERN char glb_barr_file[1024];
  #if THREAD_DEBUG >=2 
    EXTERN rnd_gen *delay_rnd_gen;
    EXTERN int *delayed_thread_barrier;
  #endif
 #endif

 #ifndef ONLY_INSTANTIATION
  bool thread_pool_locked=true;
  unsigned int nthreads=1;
 #else
  EXTERN bool thread_pool_locked;
  EXTERN unsigned int nthreads;
 #endif
  EXTERN double *glb_double_reduction_buf;
  EXTERN float_128 *glb_float_128_reduction_buf;

  EXTERN void(*threaded_function_ptr)();
#endif
  
  //endianness
  EXTERN int little_endian;

  //global input file handle
  EXTERN FILE *input_global;
  
  //vectors
  EXTERN int max_required_memory;
  EXTERN int required_memory;
  EXTERN void *main_arr;
  EXTERN nissa_vect main_vect;
  EXTERN nissa_vect *last_vect;
  EXTERN void *return_malloc_ptr;
  
  //random generator stuff
  EXTERN rnd_gen glb_rnd_gen;
  EXTERN int glb_rnd_gen_inited;
  EXTERN rnd_gen *loc_rnd_gen;
  EXTERN int loc_rnd_gen_inited;
  EXTERN enum rnd_t rnd_type_map[6]
#ifndef ONLY_INSTANTIATION
  ={RND_ALL_PLUS_ONE,RND_ALL_MINUS_ONE,RND_Z2,RND_Z2,RND_Z4,RND_GAUSS}
#endif
    ;
  EXTERN as2t smunu_entr[4];   //these are the sigma matrices entries
  EXTERN int smunu_pos[4][6];  //and positions
  
  //perpendicular dir
#ifdef ONLY_INSTANTIATION
  EXTERN int perp_dir[4][3],perp2_dir[4][3][2],perp3_dir[4][3][2];
#else
  int perp_dir[4][3]={{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
  int perp2_dir[4][3][2]={{{2,3},{1,3},{1,2}},{{2,3},{0,3},{2,3}},{{1,3},{0,3},{0,1}},{{1,2},{0,2},{0,1}}};
  int perp3_dir[4][3][2]={{{3,2},{3,1},{2,1}},{{3,2},{3,0},{3,2}},{{3,1},{3,0},{1,0}},{{2,1},{2,0},{1,0}}};
#endif
  
  //The base of the 16 gamma matrixes, the two rotators and Ci=G0*Gi*G5
  EXTERN dirac_matr base_gamma[19];
  EXTERN dirac_matr Pplus,Pminus;
  EXTERN char gtag[19][3]
#ifndef ONLY_INSTANTIATION
  ={"S0","V1","V2","V3","V0","P5","A1","A2","A3","A0","T1","T2","T3","B1","B2","B3","C1","C2","C3"}
#endif
    ;
  EXTERN int map_mu[4]
#ifndef ONLY_INSTANTIATION
  ={4,1,2,3}
#endif
    ;
  EXTERN spinspin opg[4],omg[4];
  
  EXTERN int su3_sub_gr_indices[3][2]
#ifndef ONLY_INSTANTIATION
  ={{0,1},{1,2},{0,2}}
#endif
    ;
  
  /////////////////////////////////////////// buffered comm ///////////////////////////////////
  
  EXTERN int ncomm_allocated;
  EXTERN int comm_in_prog;
  
  //buffers
  EXTERN size_t recv_buf_size,send_buf_size;
  EXTERN char *recv_buf,*send_buf;
  
  //communicators
#ifdef USE_MPI
  EXTERN comm_t lx_spin_comm,eo_spin_comm;
  EXTERN comm_t lx_color_comm,eo_color_comm;
  EXTERN comm_t lx_spincolor_comm,eo_spincolor_comm;
  EXTERN comm_t lx_spincolor_128_comm,eo_spincolor_128_comm;
  EXTERN comm_t lx_halfspincolor_comm,eo_halfspincolor_comm;
  EXTERN comm_t lx_colorspinspin_comm,eo_colorspinspin_comm;
  EXTERN comm_t lx_spinspin_comm,eo_spinspin_comm;
  EXTERN comm_t lx_su3spinspin_comm,eo_su3spinspin_comm;
  EXTERN comm_t lx_su3_comm,eo_su3_comm;
  EXTERN comm_t lx_quad_su3_comm,eo_quad_su3_comm;
#endif
  
  ////////////////////////////////////// two stage computations ///////////////////////////////
  
  EXTERN two_stage_computation_pos_t Wsklx_hopping_matrix_output_pos;
#ifdef USE_VNODES
  EXTERN two_stage_computation_pos_t virlx_hopping_matrix_output_pos;
  EXTERN two_stage_computation_pos_t viroe_or_vireo_hopping_matrix_output_pos[2];
#endif
  
  /////////////////////////////////////////// VNODES specifics ///////////////////////////////////
  
#ifdef USE_VNODES
  
  EXTERN int *virlx_of_loclx,*loclx_of_virlx;
  EXTERN int *loclx_of_vireo[2],*vireo_of_loclx;
  EXTERN int *vireo_of_loceo[2],*loceo_of_vireo[2];
  
#endif
  
  /////////////////////////////////////////// SPI specifics ///////////////////////////////////
  
#ifdef SPI
  
#include <spi/include/kernel/MU.h>
#include <spi/include/mu/InjFifo.h>
#include <spi/include/mu/GIBarrier.h>
  
  //flag to remember if spi has been initialized
  EXTERN int spi_inited;
  
  //spi rank coordinates
  EXTERN coords_5D spi_rank_coord;
  EXTERN coords_5D spi_dir_is_torus,spi_dir_size;
  
  //destination coords
  EXTERN MUHWI_Destination spi_dest[8];
  EXTERN coords_5D spi_dest_coord[8];
  
  //neighbours in the 4 dirs
  EXTERN MUHWI_Destination_t spi_neigh[2][4];
  
  //spi fifo and counters for bytes
#define nspi_fifo 8
  EXTERN uint64_t *spi_fifo[nspi_fifo],spi_desc_count[nspi_fifo];
  EXTERN MUSPI_InjFifoSubGroup_t spi_fifo_sg_ptr;
  EXTERN uint64_t spi_fifo_map[8];
  EXTERN uint8_t spi_hint_ABCD[8],spi_hint_E[8];
  
  //spi barrier
  EXTERN MUSPI_GIBarrier_t spi_barrier;
  
  //bats
  EXTERN MUSPI_BaseAddressTableSubGroup_t spi_bat_gr;
  EXTERN uint32_t spi_bat_id[2];
  
  //counter
  EXTERN volatile uint64_t spi_recv_counter;
  
  //physical address
  EXTERN uint64_t spi_send_buf_phys_addr;
  
#endif
}

#undef EXTERN
