#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "dirac_operators/stD/dirac_operator_stD.hpp"
#include "eigenvalues/eigenvalues.hpp"
#include "eigenvalues/eigenvalues_staggered.hpp"
#include "geometry/geometry_eo.hpp"
#include "geometry/geometry_mix.hpp"
#include "measures/fermions/stag.hpp"
#include "communicate/borders.hpp"

#ifdef USE_THREADS
 #include "routines/thread.hpp"
#endif

#include "spectral_measures.hpp"

namespace nissa
{

  THREADABLE_FUNCTION_3ARG(inv_participation_ratio, double *,ipratio, double *, dens, color *, v)
  {
    GET_THREAD_ID();
    double *loc_ipr=nissa_malloc("loc_ipr",glb_size[0],double);
    vector_reset(loc_ipr);
    double *loc_dens=nissa_malloc("loc_dens",glb_size[0],double);
    vector_reset(loc_dens);

    complex t;

    NISSA_PARALLEL_LOOP(loc_t,0,loc_size[0])
      for(int ivol=loc_t*loc_spat_vol;ivol<(loc_t+1)*loc_spat_vol;ivol++)
    {
        color_scalar_prod(t,v[ivol],v[ivol]);
        loc_ipr[glb_coord_of_loclx[ivol][0]]+=complex_norm2(t);
        loc_dens[glb_coord_of_loclx[ivol][0]]+=sqrt(complex_norm2(t));
    }

    double *coll_ipr=nissa_malloc("coll_ipr",glb_size[0],double);
    if(IS_MASTER_THREAD) MPI_Reduce(loc_ipr,coll_ipr,glb_size[0],MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    nissa_free(loc_ipr);

    double *coll_dens=nissa_malloc("coll_dens",glb_size[0],double);
    if(IS_MASTER_THREAD) MPI_Reduce(loc_dens,coll_dens,glb_size[0],MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    nissa_free(loc_dens);

    //normalize 
    for(int t=0;t<glb_size[0];t++)
      {
	ipratio[t]=coll_ipr[t]*glb_spat_vol;
  dens[t] = loc_dens[t];
      }
    nissa_free(coll_ipr);    
    nissa_free(coll_dens);    


  }
  THREADABLE_FUNCTION_END

  THREADABLE_FUNCTION_5ARG(chiral_components, double*,chir, quad_su3**,conf, quad_u1**, u1b, int,neigs, color **, eigvec)
  {
    color *tmpvec_eo[2]={nissa_malloc("tmpvec_eo_EVN",loc_volh+bord_volh,color),nissa_malloc("tmpvec_eo_ODD",loc_volh+bord_volh,color)};
    color *eigvec_gX_eo[2]={nissa_malloc("eigvec_gX_EVN",loc_volh+bord_volh,color),nissa_malloc("eigvec_gX_ODD",loc_volh+bord_volh,color)};
    color *eigvec_gX_lx=nissa_malloc("eigvec_gX",loc_vol+bord_vol,color);
    vector_reset(tmpvec_eo[0]);
    vector_reset(tmpvec_eo[1]);
    vector_reset(eigvec_gX_eo[0]);
    vector_reset(eigvec_gX_eo[1]);
    vector_reset(eigvec_gX_lx);
    complex tmpval;
    for(int ieig=0; ieig<neigs; ++ieig){ 
      split_lx_vector_into_eo_parts(tmpvec_eo,eigvec[ieig]);

//      // gX -> chiral condensate
//      apply_stag_op(eigvec_gX_eo,conf,u1b,stag::GAMMA_0,stag::IDENTITY,tmpvec_eo);
//      paste_eo_parts_into_lx_vector(eigvec_gX_lx,eigvec_gX_eo);
//
//      complex_vector_glb_scalar_prod(tmpval,(complex*)eigvec[ieig],(complex*)eigvec_gX_lx,loc_vol*sizeof(color)/sizeof(complex));
//      chir[0][ieig] = tmpval[RE];
//      master_printf("chircond(%d) = %.16lg\t%.16lg\n",ieig,tmpval[RE],tmpval[IM]);
//
//      vector_reset(eigvec_gX_eo[0]);
//      vector_reset(eigvec_gX_eo[1]);
//      vector_reset(eigvec_gX_lx);
//      tmpval[RE]=0.0;
//      tmpval[IM]=0.0;

      // g5 -> chirality
      apply_stag_op(eigvec_gX_eo,conf,u1b,stag::GAMMA_5,stag::IDENTITY,tmpvec_eo);
      paste_eo_parts_into_lx_vector(eigvec_gX_lx,eigvec_gX_eo);

      complex_vector_glb_scalar_prod(tmpval,(complex*)eigvec[ieig],(complex*)eigvec_gX_lx,loc_vol*sizeof(color)/sizeof(complex));
      chir[ieig] = tmpval[RE];
    }
    nissa_free(tmpvec_eo[EVN]);
    nissa_free(tmpvec_eo[ODD]);
    nissa_free(eigvec_gX_eo[EVN]);
    nissa_free(eigvec_gX_eo[ODD]);
    nissa_free(eigvec_gX_lx);
  }
  THREADABLE_FUNCTION_END

  // This measure will compute the first 'n' eigenvalues (parameter)
  // and eigenvectors of the iD operator in staggered formulation, in order to
  // build an estimate of the topological susceptibility.
  // refs:  https://arxiv.org/pdf/1008.0732.pdf for the susceptibility formula,
  //        https://arxiv.org/pdf/0912.2850.pdf for the 2^(d/2) overcounting.
  THREADABLE_FUNCTION_8ARG(measure_iDst_spectrum, color**,eigvec, quad_su3**,conf, double *, chir, complex*, eigval, int,neigs, bool, minmax, double,eig_precision, int,wspace_size)
  {
    //identity backfield
    quad_u1 *u1b[2]={nissa_malloc("u1b",loc_volh+bord_volh,quad_u1),nissa_malloc("u1b",loc_volh+bord_volh,quad_u1)};
    init_backfield_to_id(u1b);
    add_antiperiodic_condition_to_backfield(u1b,0);

    //launch the eigenfinder
    
    double eig_time=-take_time();
    find_eigenvalues_staggered_iD(eigvec,eigval,neigs,minmax,conf,u1b,eig_precision,wspace_size);
    eig_time+=take_time();
    verbosity_lv1_master_printf("Eigenvalues time: %lg\n",eig_time);

    verbosity_lv2_master_printf("\n\nEigenvalues of staggered iD operator:\n");
    chiral_components(chir, conf, u1b, neigs, eigvec);

    for(int ieig=0;ieig<neigs;ieig++)
    {
      verbosity_lv2_master_printf("lam_%d = (%.16lg,%.16lg)\n",ieig,eigval[ieig][RE],eigval[ieig][IM]);
    }
    verbosity_lv2_master_printf("\n\n\n");
    
    
    nissa_free(u1b[0]);
    nissa_free(u1b[1]);
  }
  THREADABLE_FUNCTION_END



  //measure of spectrum of chosen operator
  void measure_spectral_props(quad_su3 **conf,theory_pars_t &theory_pars,spectr_props_meas_pars_t &meas_pars,int iconf,bool conf_created)
  {

    int neigs = meas_pars.neigs;
    std::string opname = meas_pars.opname;
    // allocate auxiliary vectors 
    complex *eigval=nissa_malloc("eigval",neigs,complex);
    vector_reset(eigval);
    double part_ratios[neigs*glb_size[0]];
    double dens4slice[neigs*glb_size[0]];

    
    color **eigvec=nissa_malloc("eigvec",neigs,color*);
    for(int ieig=0;ieig<neigs;ieig++){
      eigvec[ieig]=nissa_malloc("eigvec_ieig",loc_vol+bord_vol,color);
      vector_reset(eigvec[ieig]);
    }


//    double * chir[2] = {nissa_malloc("chir_cond",neigs,double),nissa_malloc("chir_ality",neigs,double)};
    double * chir = nissa_malloc("chir",neigs,double);
    vector_reset(chir);

    verbosity_lv1_master_printf("Measuring spectrum for %s\n",opname.c_str());
    measure_iDst_spectrum(eigvec,conf,chir,eigval,neigs,meas_pars.minmax,meas_pars.eig_precision,meas_pars.wspace_size);


    for(int ieig=0;ieig<neigs;ieig++){
      inv_participation_ratio(&part_ratios[ieig*glb_size[0]],&dens4slice[ieig*glb_size[0]],eigvec[ieig]);
    }


    //print the result on file
    FILE *file=open_file(meas_pars.path,conf_created?"w":"a");

    master_fprintf(file,"%d\t%d\t%d\t",iconf,meas_pars.smooth_pars.nsmooth(),neigs);
    for(int ieig=0;ieig<neigs;++ieig)
      master_fprintf(file,"%.16lg\t",eigval[ieig][RE]);
    for(int i=0;i<neigs*glb_size[0];++i){
      master_fprintf(file,"%.16lg\t",part_ratios[i]);
    }
    for(int ieig=0; ieig<neigs; ++ieig){
      master_fprintf(file,"%.16lg\t",chir[ieig]);
    }
    for(int i=0;i<neigs*glb_size[0];++i){
      master_fprintf(file,"%.16lg\t",dens4slice[i]);
    }
    master_fprintf(file,"\n");
    
    close_file(file);
   
    // deallocating vectors 

    for(int ieig=0;ieig<neigs;ieig++)
        nissa_free(eigvec[ieig]);
    nissa_free(eigvec);
    nissa_free(eigval);
    nissa_free(chir);
  }
  
  //print pars
  std::string spectr_props_meas_pars_t::get_str(bool full)
  {
    std::ostringstream os;
    
    os<<"MeasSpectrProps\n";
    os<<base_fermionic_meas_t::get_str(full);
    if(opname!=def_opname() or full) os<<" OPName\t\t=\t"<<opname<<"\n";
    if(neigs!=def_neigs() or full) os<<" Neigs\t\t=\t"<<neigs<<"\n";
    if(minmax!=def_minmax() or full) os<<" MinMax\t\t=\t"<<minmax<<"\n";
    if(eig_precision!=def_eig_precision() or full) os<<" EigPrecision\t\t=\t"<<eig_precision<<"\n";
    if(wspace_size!=def_wspace_size() or full) os<<" WSpaceSize\t\t=\t"<<wspace_size<<"\n";
    os<<smooth_pars.get_str(full);
    
    return os.str();
  }
}
