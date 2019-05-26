#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "dirac_operators/stD/dirac_operator_stD.hpp"
#include "eigenvalues/eigenvalues.hpp"
#include "eigenvalues/eigenvalues_staggered.hpp"
#include "geometry/geometry_eo.hpp"
#include "geometry/geometry_mix.hpp"
#include "measures/fermions/stag.hpp"

#ifdef USE_THREADS
 #include "routines/thread.hpp"
#endif

#include "spectral_measures.hpp"

namespace nissa
{

  //Computes the participation ratio on the time slices
  void participation_ratio(double *pratios, color *v)
  {
    
    for(int ti=0; ti<glb_size[0]; ++ti)
    {
      
      GET_THREAD_ID();
      
      double *l=nissa_malloc("l",glb_spat_vol,double);
      
      NISSA_PARALLEL_LOOP(ivol,0,glb_spat_vol)
      {
        complex t;

        int ilx = ti+glb_size[0]*ivol;


        color_scalar_prod(t,v[ilx],v[ilx]);
        l[ilx]=t[RE];
      }
      THREAD_BARRIER();
      
      double s=double_vector_glb_norm2(l,glb_spat_vol);
      double n2=double_vector_glb_norm2(v,glb_spat_vol);
      
      pratios[ti]=sqr(n2)/(glb_spat_vol*s);

      nissa_free(l);
    }    
  }

  void chiral_components(double **chir, quad_su3 **conf, quad_u1 **u1b, int neigs, color **eigvec){
    //identity backfield

    color *tmpvec_eo[2]={nissa_malloc("tmpvec_eo_EVN",loc_volh+bord_volh,color),nissa_malloc("tmpvec_eo_ODD",loc_volh+bord_volh,color)};
    color *eigvec_gX_eo[2]={nissa_malloc("eigvec_gX_EVN",loc_volh+bord_volh,color),nissa_malloc("eigvec_gX_ODD",loc_volh+bord_volh,color)};
    color *eigvec_gX_lx=nissa_malloc("eigvec_gX",loc_vol+bord_vol,color);
   complex tmpval;
   for(int ieig=0; ieig<neigs; ++ieig){ 
      split_lx_vector_into_eo_parts(tmpvec_eo,eigvec[ieig]);

      // gX -> chiral condensate
      apply_stag_op(eigvec_gX_eo,conf,u1b,stag::GAMMA_0,stag::IDENTITY,tmpvec_eo);
      paste_eo_parts_into_lx_vector(eigvec_gX_lx,eigvec_gX_eo);
      
      complex_vector_glb_scalar_prod(tmpval,(complex*)eigvec[ieig],(complex*)eigvec_gX_lx,loc_vol*sizeof(color)/sizeof(complex));
      chir[0][ieig] = tmpval[RE];
      
      // g5 -> chirality
      apply_stag_op(eigvec_gX_eo,conf,u1b,stag::GAMMA_5,stag::IDENTITY,tmpvec_eo);
      paste_eo_parts_into_lx_vector(eigvec_gX_lx,eigvec_gX_eo);
      
      complex_vector_glb_scalar_prod(tmpval,(complex*)eigvec[ieig],(complex*)eigvec_gX_lx,loc_vol*sizeof(color)/sizeof(complex));
      chir[1][ieig] = tmpval[RE];
    }
    nissa_free(tmpvec_eo[EVN]);
    nissa_free(tmpvec_eo[ODD]);
    nissa_free(eigvec_gX_eo[EVN]);
    nissa_free(eigvec_gX_eo[ODD]);
    nissa_free(eigvec_gX_lx);
  }

  // This measure will compute the first 'n' eigenvalues (parameter)
  // and eigenvectors of the iD operator in staggered formulation, in order to
  // build an estimate of the topological susceptibility.
  // refs:  https://arxiv.org/pdf/1008.0732.pdf for the susceptibility formula,
  //        https://arxiv.org/pdf/0912.2850.pdf for the 2^(d/2) overcounting.
  THREADABLE_FUNCTION_9ARG(measure_iDst_spectrum, color**,eigvec, quad_su3**,conf, complex*, eigval, double**, part_ratios, double **, chir, int,neigs, bool, minmax, double,eig_precision, int,wspace_size)
  {
    //identity backfield
    quad_u1 *u1b[2]={nissa_malloc("u1b",loc_volh+bord_volh,quad_u1),nissa_malloc("u1b",loc_volh+bord_volh,quad_u1)};
    init_backfield_to_id(u1b);
    add_antiperiodic_condition_to_backfield(u1b,0);

    //launch the eigenfinder
    double eig_time=-take_time();
    find_eigenvalues_staggered_iD(eigvec,eigval,neigs,minmax,conf,u1b,eig_precision,wspace_size);
    
    verbosity_lv2_master_printf("\n\nEigenvalues of staggered iD operator:\n");
    for(int ieig=0;ieig<neigs;ieig++)
    {
      participation_ratio(part_ratios[ieig],eigvec[ieig]);
    }
    chiral_components(chir, conf, u1b, neigs, eigvec);

    for(int ieig=0;ieig<neigs;ieig++)
    {
      verbosity_lv2_master_printf("lam_%d = (%.16lg,%.16lg)\n",ieig,eigval[ieig][RE],eigval[ieig][IM]);
    }
    verbosity_lv2_master_printf("\n\n\n");
    
    eig_time+=take_time();
    verbosity_lv1_master_printf("Eigenvalues time: %lg\n",eig_time);
    
    nissa_free(u1b[0]);
    nissa_free(u1b[1]);
  }
  THREADABLE_FUNCTION_END

//  THREADABLE_FUNCTION_8ARG(measure_iDov_spectrum, spincolor**,eigvec, quad_su3*,conf,complex*, eigval, int,neigs, double*, part_ratios, bool, minmax, double,eig_precision, int,wspace_size)
//  {
//    //launch the eigenfinder
//    double eig_time=-take_time();
//    find_eigenvalues_overlap(eigvec,eigval,neigs,minmax,conf,eig_precision,wspace_size);
//    
//    verbosity_lv2_master_printf("\n\nEigenvalues of staggered iD operator:\n");
//    for(int ieig=0;ieig<neigs;ieig++)
//    {
//      verbosity_lv2_master_printf("lam_%d = (%.16lg,%.16lg)\n",ieig,eigval[ieig][RE],eigval[ieig][IM]);
//    }
//    verbosity_lv2_master_printf("\n\n\n");
//    
//    eig_time+=take_time()
//    verbosity_lv1_master_printf("Eigenvalues time: %lg\n",eig_time);
//    
//    nissa_free(tmpvec_eo[EVN]);
//    nissa_free(tmpvec_eo[ODD]);
//  }
//  THREADABLE_FUNCTION_END


  //measure of spectrum of chosen operator
  void measure_spectral_props(quad_su3 **conf,theory_pars_t &theory_pars,spectr_props_meas_pars_t &meas_pars,int iconf,bool conf_created)
  {

    int neigs = meas_pars.neigs;
    std::string opname = meas_pars.opname;

    // allocate auxiliary vectors 
    quad_su3 *conf_lx=nissa_malloc("conf_lx",loc_vol+bord_vol,quad_su3);
    if(opname=="iDst"){
      ;
    }else if(opname=="iDov"){
      paste_eo_parts_into_lx_vector(conf_lx,conf);  
    }
    complex *eigval=nissa_malloc("eigval",neigs,complex);
    double **part_ratios=nissa_malloc("part_ratios",neigs,double*);
    for(int ieig=0; ieig<neigs; ieig++)
      part_ratios[ieig] = nissa_malloc("part_ratios_ieig",glb_size[0],double);
    
    color **eigvec_col;
    spincolor **eigvec_spincol;
    if(opname=="iDst"){
      eigvec_col=nissa_malloc("eigvec",neigs,color*);
      for(int ieig=0;ieig<neigs;ieig++)
        eigvec_col[ieig]=nissa_malloc("eigvec_col_ieig",loc_vol+bord_vol,color);
    }else if(opname=="iDov"){
      eigvec_spincol=nissa_malloc("eigvec_lx",neigs,spincolor*);
      for(int ieig=0;ieig<neigs;ieig++)
        eigvec_spincol[ieig]=nissa_malloc("eigvec_spincol_ieig",loc_vol+bord_vol,spincolor);
    }
    double * chir[2] = {nissa_malloc("chir_cond",neigs,double),nissa_malloc("chir_ality",neigs,double)};
    

    //loop on smooth
    int nsmooth=0;
    bool finished;
    do
      { 
      verbosity_lv1_master_printf("Measuring spectrum for %s with nsmooth %d/%d\n",opname.c_str(), nsmooth, meas_pars.smooth_pars.nsmooth());

      // reset vectors
      vector_reset(eigval);
      if(opname=="iDst"){
        for(int ieig=0;ieig<neigs;ieig++)
          vector_reset(eigvec_col[ieig]);
      }else if(opname=="iDwil"){
        for(int ieig=0;ieig<neigs;ieig++)
          vector_reset(eigvec_spincol[ieig]);
      }
      if(opname=="iDst"){ 
        measure_iDst_spectrum(eigvec_col,conf,eigval,part_ratios,chir, neigs,meas_pars.minmax,meas_pars.eig_precision,meas_pars.wspace_size);
      }else if (opname=="iDov"){
        ;//measure_iDov_spectrum(eigvec_spincol,conf_lx,eigval,part_ratios,neigs,meas_pars.minmax,meas_pars.eig_precision,meas_pars.wspace_size);
      }

    //print the result on file
    FILE *file=open_file(meas_pars.path,conf_created?"w":"a");

    master_fprintf(file,"%d\t%d\t%d\t",iconf,meas_pars.smooth_pars.nsmooth(),neigs);
    for(int ieig=0;ieig<neigs;++ieig)
      master_fprintf(file,"%.16lg\t",eigval[ieig][RE]);
    for(int ieig=0;ieig<neigs*glb_size[0];++ieig) for(int ti=0; ti<glb_size[0]; ++ti){
      master_fprintf(file,"%.16lg\t",part_ratios[ieig][ti]);
    }
    for(int ieig=0; ieig<neigs; ++ieig){
      master_fprintf(file,"%.16lg\t",chir[0][ieig]);
    }
    for(int ieig=0; ieig<neigs; ++ieig){
      master_fprintf(file,"%.16lg\t",chir[1][ieig]);
    }
    master_fprintf(file,"\n");
    
    close_file(file);

    //proceeds with smoothing
    if(opname=="iDst"){
      paste_eo_parts_into_lx_vector(conf_lx,conf);
    }
    finished=smooth_lx_conf_until_next_meas(conf_lx,meas_pars.smooth_pars,nsmooth);
    if(opname=="iDst"){
      split_lx_vector_into_eo_parts(conf,conf_lx);
    }

  }
  while(not finished);
   
    // deallocating vectors 

    if(opname=="iDst"){
      for(int ieig=0;ieig<neigs;ieig++)
          nissa_free(eigvec_col[ieig]);
      nissa_free(eigvec_col);
    }else if(opname=="iDov"){
      for(int ieig=0;ieig<neigs;ieig++)
          nissa_free(eigvec_spincol[ieig]);
      nissa_free(eigvec_spincol);
    }
    nissa_free(eigval);
    nissa_free(chir[0]);
    nissa_free(chir[1]);
    for(int ieig=0; ieig<neigs; ieig++)
      nissa_free(part_ratios[ieig]);
		nissa_free(part_ratios);
    nissa_free(conf_lx);
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
