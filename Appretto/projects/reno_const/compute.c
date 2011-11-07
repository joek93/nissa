#include "appretto.h"
#include "../semileptonic/addendum.c"

//gauge info
int ngauge_conf;
char conf_path[1024];
quad_su3 *conf,*unfix_conf;
double kappa;

int nmass;
double *mass;
double put_theta[4],old_theta[4]={0,0,0,0};

//source data
int source_coord[4]={0,0,0,0};
spincolor *source;
su3spinspin *original_source,*original_source_fft;

//vectors for the spinor data
int npropS0;
su3spinspin **S0[2];
spincolor **cgmms_solution,*reco_solution[2];

//cgmms inverter parameters
double stopping_residue;
double minimal_residue;
int stopping_criterion;
int niter_max;

//two points contractions
int ncontr_2pts;
complex *contr_2pts;
int *op1_2pts,*op2_2pts;
char outfile_2pts[1024];

//timings
int ninv_tot=0,ncontr_tot=0;
double tot_time=0,inv_time=0,contr_time=0,fix_time=0;

//This function takes care to make the revert on the FIRST spinor, putting the needed gamma5
void meson_two_points(complex *corr,int *list_op1,su3spinspin *s1,int *list_op2,su3spinspin *s2,int ncontr)
{
  //Temporary vectors for the internal gamma
  dirac_matr t1[ncontr],t2[ncontr];

  for(int icontr=0;icontr<ncontr;icontr++)
    {
      //Put the two gamma5 needed for the revert of the first spinor
      dirac_prod(&(t1[icontr]), &(base_gamma[list_op1[icontr]]),&(base_gamma[5]));
      dirac_prod(&(t2[icontr]), &(base_gamma[5]),&(base_gamma[list_op2[icontr]]));
    }
  
  //Call the routine which does the real contraction
  trace_g_ccss_dag_g_ccss(corr,t1,s1,t2,s2,ncontr);
}

//Generate the source for the dirac index
void generate_source()
{ //reset
  memset(original_source,0,sizeof(su3spinspin)*loc_vol);

  int islocal=1,lx[4];
  for(int idir=0;idir<4;idir++)
    {
      lx[idir]=source_coord[idir]-proc_coord[idir]*loc_size[idir];
      islocal&=(lx[idir]>=0);
      islocal&=(lx[idir]<loc_size[idir]);
    }
  
  if(islocal)
    for(int id=0;id<4;id++)
      for(int ic=0;ic<3;ic++)
	original_source[loclx_of_coord(lx)][ic][ic][id][id][0]=1;
}

//Parse all the input file
void initialize_semileptonic(char *input_path)
{
  open_input(input_path);

  // 1) Read information about the gauge conf
  
  //Read the volume
  read_str_int("L",&(glb_size[1]));
  read_str_int("T",&(glb_size[0]));
  //Init the MPI grid 
  init_grid(); 
  //Kappa
  read_str_double("Kappa",&kappa);
  
  // 2) Read information about the masses
  
  //Read the masses
  read_list_of_doubles("NMass",&nmass,&mass);

  // 3) Info about the inverter

  //Residue
  read_str_double("Residue",&stopping_residue);
  //Stopping criterion
  stopping_criterion=numb_known_stopping_criterion;
  char str_stopping_criterion[1024];
  read_str_str("StoppingCriterion",str_stopping_criterion,1024);
  int isc=0;
  do
    {
      if(strcasecmp(list_known_stopping_criterion[isc],str_stopping_criterion)==0) stopping_criterion=isc;
      isc++;
    }
  while(isc<numb_known_stopping_criterion && stopping_criterion==numb_known_stopping_criterion);
  
  if(stopping_criterion==numb_known_stopping_criterion && rank==0)
    {
      fprintf(stderr,"Unknown stopping criterion: %s\n",str_stopping_criterion);
      fprintf(stderr,"List of known stopping criterions:\n");
      for(int isc=0;isc<numb_known_stopping_criterion;isc++) fprintf(stderr," %s\n",list_known_stopping_criterion[isc]);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  if(stopping_criterion==sc_standard) read_str_double("MinimalResidue",&minimal_residue);
      
  //Number of iterations
  read_str_int("NiterMax",&niter_max);
  
  // 4) contraction list for two points
  
  read_str_int("NContrTwoPoints",&ncontr_2pts);
  contr_2pts=appretto_malloc("contr_2pts",ncontr_2pts*glb_size[0],complex);
  op1_2pts=appretto_malloc("op1_2pts",ncontr_2pts,int);
  op2_2pts=appretto_malloc("op2_2pts",ncontr_2pts,int);
  for(int icontr=0;icontr<ncontr_2pts;icontr++)
    {
      //Read the operator pairs
      read_int(&(op1_2pts[icontr]));
      read_int(&(op2_2pts[icontr]));

      master_printf(" contr.%d %d %d\n",icontr,op1_2pts[icontr],op2_2pts[icontr]);
    }
  
  read_str_int("NGaugeConf",&ngauge_conf);  
  
  ////////////////////////////////////// end of input reading/////////////////////////////////
  
  //allocate gauge conf and all the needed spincolor and su3spinspin
  conf=appretto_malloc("or_conf",loc_vol+loc_bord,quad_su3);
  unfix_conf=appretto_malloc("unfix_conf",loc_vol+loc_bord,quad_su3);
  
  //Allocate all the S0 su3spinspin vectors
  npropS0=nmass;
  S0[0]=appretto_malloc("S0[0]",npropS0,su3spinspin*);
  S0[1]=appretto_malloc("S0[1]",npropS0,su3spinspin*);
  for(int iprop=0;iprop<npropS0;iprop++)
    {
      S0[0][iprop]=appretto_malloc("S0[0][iprop]",loc_vol,su3spinspin);
      S0[1][iprop]=appretto_malloc("S0[1][iprop]",loc_vol,su3spinspin);
    }
  
  //Allocate nmass spincolors, for the cgmms solutions
  cgmms_solution=appretto_malloc("cgmms_solution",nmass,spincolor*);
  for(int imass=0;imass<nmass;imass++) cgmms_solution[imass]=appretto_malloc("cgmms_solution[imass]",loc_vol+loc_bord,spincolor);
  reco_solution[0]=appretto_malloc("reco_sol[0]",loc_vol+loc_bord,spincolor);
  reco_solution[1]=appretto_malloc("reco_sol[1]",loc_vol+loc_bord,spincolor);
  
  //Allocate one spincolor for the source
  source=appretto_malloc("source",loc_vol+loc_bord,spincolor);
  original_source=appretto_malloc("orig_source",loc_vol,su3spinspin);
  original_source_fft=appretto_malloc("orig_source_fft",loc_vol+loc_bord,su3spinspin);
}

//load the conf, fix it and put boundary cond
void load_gauge_conf()
{
  //load the gauge conf, propagate borders, calculate plaquette and PmuNu term
  read_gauge_conf(unfix_conf,conf_path);
  //prepare the fixed version and calculate plaquette
  fix_time-=take_time();
  landau_gauge_fix(conf,unfix_conf,1.e-20);
  //landau_gauge_fix(conf,unfix_conf,1.e-20);
  //memcpy(conf,unfix_conf,loc_vol*sizeof(quad_su3));
  fix_time+=take_time();
  master_printf("Fixed conf in %lg sec\n",fix_time);
  communicate_gauge_borders(conf);
  communicate_gauge_borders(unfix_conf);
  
  double gplaq=global_plaquette(conf);
  if(rank==0) printf("plaq: %.18g\n",gplaq);
  gplaq=global_plaquette(unfix_conf);
  if(rank==0) printf("unfix plaq: %.18g\n",gplaq);

  //Put the anti-periodic condition on the temporal border
  old_theta[0]=0;
  put_theta[0]=1;
  adapt_theta(conf,old_theta,put_theta,1,0);
}

//Finalization
void close_semileptonic()
{
  appretto_free(original_source);appretto_free(original_source_fft);appretto_free(source);
  appretto_free(reco_solution[1]);appretto_free(reco_solution[0]);
  for(int imass=0;imass<nmass;imass++) appretto_free(cgmms_solution[imass]);
  appretto_free(cgmms_solution);
  for(int r=0;r<2;r++)
    {
      for(int iprop=0;iprop<npropS0;iprop++)
	appretto_free(S0[r][iprop]);
      appretto_free(S0[r]);
    }
  appretto_free(unfix_conf);
  appretto_free(conf);
  appretto_free(contr_2pts);
  appretto_free(op1_2pts);
  appretto_free(op2_2pts);

  if(rank==0)
    {
      printf("\n");
      printf("Total time: %g, of which:\n",tot_time);
      printf(" - %02.2f%s to fix %d conf. (%2.2gs avg)\n",fix_time/tot_time*100,"%",ngauge_conf,fix_time/ngauge_conf);
      printf(" - %02.2f%s to perform %d inversions (%2.2gs avg)\n",inv_time/tot_time*100,"%",ninv_tot,inv_time/ninv_tot);
      printf(" - %02.2f%s to perform %d contr. (%2.2gs avg)\n",contr_time/tot_time*100,"%",ncontr_tot,contr_time/ncontr_tot);
    }
  
  close_appretto();
}


void remove_anti_periodic_cond(spincolor *S)
{
  for(int ivol=0;ivol<loc_vol;ivol++)
    {
      int dt=(glb_coord_of_loclx[ivol][0]+source_coord[0])%glb_size[0]-source_coord[0];
      double arg=M_PI*dt/glb_size[0];
      complex phase={cos(arg),sin(arg)};
      
      safe_spincolor_prod_complex(S[ivol],S[ivol],phase);
    }
}


//calculate the standard propagators
void calculate_S0()
{
  for(int id=0;id<4;id++)
    for(int ic=0;ic<3;ic++)
      { //loop over the source dirac index
	for(int ivol=0;ivol<loc_vol;ivol++)
	  {
	    get_spincolor_from_su3spinspin(source[ivol],original_source[ivol],id,ic);
	    //put the g5
	    for(int id1=2;id1<4;id1++) for(int ic1=0;ic1<3;ic1++) for(int ri=0;ri<2;ri++) source[ivol][id1][ic1][ri]*=-1;
	  }
	
	double part_time=-take_time();
	communicate_lx_spincolor_borders(source);
	inv_Q2_cgmms(cgmms_solution,source,NULL,conf,kappa,mass,nmass,niter_max,stopping_residue,minimal_residue,stopping_criterion);
	part_time+=take_time();ninv_tot++;inv_time+=part_time;
	master_printf("Finished the inversion of S0, dirac index %d, color %d in %g sec\n",id,ic,part_time);
	
	for(int imass=0;imass<nmass;imass++)
	  { //reconstruct the doublet
	    reconstruct_doublet(reco_solution[0],reco_solution[1],cgmms_solution[imass],conf,kappa,mass[imass]);
	    
	    //remove_anti_periodic_cond(reco_solution[0]);
	    //remove_anti_periodic_cond(reco_solution[1]);
	    
	    if(rank==0) printf("Mass %d (%g) reconstructed \n",imass,mass[imass]);
	    for(int r=0;r<2;r++) //convert the id-th spincolor into the su3spinspin
	      for(int i=0;i<loc_vol;i++)
		put_spincolor_into_su3spinspin(S0[r][imass][i],reco_solution[r][i],id,ic);
	  }
      }
  
  for(int r=0;r<2;r++) //remember that D^-1 rotate opposite than D!
    for(int ipropS0=0;ipropS0<npropS0;ipropS0++) //put the (1+ig5)/sqrt(2) factor
      {
	rotate_vol_su3spinspin_to_physical_basis(S0[r][ipropS0],!r,!r);
	char path[1024];
	sprintf(path,"out_r%1d_im%02d",r,ipropS0);
	write_su3spinspin(path,S0[r][ipropS0],64);
      }
}

//compute the fft of the source
void compute_source_fft(double sign)
{fft4d((complex*)original_source_fft,(complex*)original_source,144,sign,0);}

//compute the fft of all propagators
void compute_fft(double sign)
{
  for(int r=0;r<2;r++)
    for(int imass=0;imass<nmass;imass++)
      {
	fft4d((complex*)S0[r][imass],(complex*)S0[r][imass],144,sign,0);
	//multiply by the hermitean of the fft of the source
	for(int imom=0;imom<loc_vol;imom++)
	  for(int ic_so=0;ic_so<3;ic_so++)
	    for(int id_so=0;id_so<4;id_so++)
	      for(int ic_si=0;ic_si<3;ic_si++)
		for(int id_si=0;id_si<4;id_si++)
		  safe_complex_conj1_prod(S0[r][imass][imom][ic_si][ic_so][id_si][id_so],
					  S0[r][imass][imom][ic_si][ic_so][id_si][id_so],
					  original_source_fft[imom][ic_so][ic_si][id_so][id_si]);
      }
}

//filter the computed momentum
void print_momentum_subset()
{
  FILE *fout=(rank==0) ? fopen("fft_internal","w") : NULL;
  int glb_ip[4];
  int interv[2][2][2]={{{0,3},{0,2}},{{4,7},{2,3}}};
  for(int imass=0;imass<nmass;imass++)
    for(int r=0;r<2;r++)
      for(int iinterv=0;iinterv<2;iinterv++)
	{
	  for(glb_ip[0]=interv[iinterv][0][0];glb_ip[0]<=interv[iinterv][0][1];glb_ip[0]++)
	    for(glb_ip[1]=interv[iinterv][1][0];glb_ip[1]<=interv[iinterv][1][1];glb_ip[1]++)
	      for(glb_ip[2]=interv[iinterv][1][0];glb_ip[2]<=interv[iinterv][1][1];glb_ip[2]++)
		for(glb_ip[3]=interv[iinterv][1][0];glb_ip[3]<=interv[iinterv][1][1];glb_ip[3]++)
		  {
		    //check locality and identify local element
		    int loc_ip[4],isloc=1;
		    for(int mu=0;mu<4;mu++)
		      {
			loc_ip[mu]=glb_ip[mu]-loc_size[mu]*proc_coord[mu];
			isloc=isloc & (glb_ip[mu]/loc_size[mu]==proc_coord[mu]);
		      }
		    
		    //if it is local print it
		    if(isloc)
		      {
			int p=loclx_of_coord(loc_ip);
			for(int ic_so=0;ic_so<3;ic_so++)
			  for(int id_so=0;id_so<4;id_so++)
			    for(int ic_si=0;ic_si<3;ic_si++)
			      for(int id_si=0;id_si<4;id_si++)
				{
				  /*
				    printf("%d ",rank);
				    for(int mu=0;mu<4;mu++) printf("%d ",glb_ip[mu]);
				    printf("im=%d r=%d ic_si=%d ic_so=%d id_si=%d id_so=%d  %lg %lg\n",
				    imass,r,ic_si,ic_so,id_si,id_so,
				    S0[r][imass][p][ic_si][ic_so][id_si][id_so][0],S0[r][imass][p][ic_si][ic_so][id_si][id_so][1]);
				  */
				  fwrite(S0[r][imass][p][ic_si][ic_so][id_si][id_so],sizeof(double),2,fout);
				}
		      }
		    fflush(fout);
		    MPI_Barrier(MPI_COMM_WORLD);
		  }
	}
}

//Calculate and print to file the 2pts
void calculate_all_2pts()
{
  FILE *fout=open_text_file_for_output(outfile_2pts);
  
  //perform the contractions
  contr_time-=take_time();      
  for(int im2=0;im2<nmass;im2++)
    for(int r2=0;r2<2;r2++)
      for(int im1=0;im1<nmass;im1++)
	for(int r1=0;r1<2;r1++)
	  {
	    master_fprintf(fout," # m1=%f r1=%d , m2=%f r2=%d \n",mass[im1],r1,mass[im2],r2);
	    
	    meson_two_points(contr_2pts,op1_2pts,S0[r1][im1],op2_2pts,S0[r2][im2],ncontr_2pts);
	    ncontr_tot+=ncontr_2pts;
	    print_contractions_to_file_ptv(fout,ncontr_2pts,op1_2pts,op2_2pts,contr_2pts,source_coord[0],"");
	    
	    if(rank==0) fprintf(fout,"\n");
	  }
  contr_time+=take_time();
  
}

int main(int narg,char **arg)
{
  //Basic mpi initialization
  init_appretto();
  
  if(narg<2) crash("Use: %s input_file",arg[0]);
  
  tot_time-=take_time();
  initialize_semileptonic(arg[1]);
  
  for(int iconf=0;iconf<ngauge_conf;iconf++)
  {
    //Gauge path
      read_str(conf_path,1024);
      read_int(&(source_coord[0]));
      read_int(&(source_coord[1]));
      read_int(&(source_coord[2]));
      read_int(&(source_coord[3]));
      read_str(outfile_2pts,1024);
      
      load_gauge_conf();
      generate_source();
      compute_source_fft(-1);
      
      calculate_S0();
      calculate_all_2pts();

      compute_fft(-1);
      print_momentum_subset();
  }
  
  tot_time+=take_time();
  close_semileptonic();
  
  return 0;
}
