#ifndef _su3_h
#define _su3_h
double real_part_of_trace_su3_prod_su3_dag(su3 a,su3 b);
double su2_part_of_su3(double &A,double &B,double &C,double &D,su3 in,int isub_gr);
double su2_part_of_su3(su2 out,su3 in,int isub_gr);
double su3_normq(su3 U);
void as2t_su3_put_to_zero(as2t_su3 m);
void color_copy(color b,color a);
void color_isubt(color a,color b,color c);
void color_isubtassign(color a,color b);
void color_isumm(color a,color b,color c);
void color_isummassign(color a,color b);
void color_print(color c);
void color_prod_double(color a,color b,double c);
void color_put_to_gauss(color H,rnd_gen *gen,double sigma);
void color_put_to_zero(color m);
void color_subt(color a,color b,color c);
void color_subtassign(color a,color b);
void color_summ(color a,color b,color c);
void color_summassign(color a,color b);
void herm_put_to_gauss(su3 H,rnd_gen *gen,double sigma);
void quad_su3_copy(quad_su3 b,quad_su3 a);
void safe_color_prod_complex(color out,color in,complex factor);
void safe_color_prod_complex_conj(color out,color in,complex factor);
void safe_color_prod_su3(color a,color b,su3 c);
void safe_dirac_prod_spincolor(spincolor out,dirac_matr *m,spincolor in);
void safe_spincolor_prod_complex(spincolor out,spincolor in,complex factor);
void safe_spincolor_prod_dirac(spincolor out,spincolor in,dirac_matr *m);
void safe_spincolor_summ_with_cfactor(spincolor a,spincolor b,spincolor c,complex factor);
void safe_su3_dag_prod_color(color a,su3 b,color c);
void safe_su3_dag_prod_spincolor(spincolor out,su3 U,spincolor in);
void safe_su3_dag_prod_su3(su3 a,su3 b,su3 c);
void safe_su3_dag_prod_su3_dag(su3 a,su3 b,su3 c);
void safe_su3_explicit_inverse(su3 invU,su3 U);
void safe_su3_hermitian(su3 out,su3 in);
void safe_su3_prod_color(color a,su3 b,color c);
void safe_su3_prod_complex(su3 a,su3 b,complex c);
void safe_su3_prod_conj_complex(su3 a,su3 b,complex c);
void safe_su3_prod_su3(su3 a,su3 b,su3 c);
void safe_su3_prod_su3_dag(su3 a,su3 b,su3 c);
void safe_su3spinspin_prod_complex(su3spinspin out,su3spinspin in,complex factor);
void spincolor_copy(spincolor b,spincolor a);
void spincolor_print(spincolor c);
void spincolor_prod_double(spincolor out,spincolor in,double factor);
void spincolor_put_to_zero(spincolor m);
void spincolor_subt(spincolor a,spincolor b,spincolor c);
void spincolor_subtassign(spincolor a,spincolor b);
void spincolor_summ(spincolor a,spincolor b,spincolor c);
void spincolor_summ_the_prod_complex(spincolor out,spincolor in,complex factor);
void spincolor_summ_the_prod_double(spincolor a,spincolor b,spincolor c,double factor);
void spincolor_summassign(spincolor a,spincolor b);
void su2_prodassign_su3(double A,double B,double C,double D,int isub_gr,su3 in);
void su2_prodassign_su3(su2 mod,int isub_gr,su3 in);
void su3_copy(su3 b,su3 a);
void quad_su3_nissa_to_ildg_reord(quad_su3 out,quad_su3 in);
void quad_su3_ildg_to_nissa_reord(quad_su3 out,quad_su3 in);
void su3_dag_subt_the_prod_color(color a,su3 b,color c);
void su3_dag_summ_the_prod_color(color a,su3 b,color c);
void su3_det(complex d,su3 U);
void su3_find_cooled(su3 u,quad_su3 **eo_conf,int par,int ieo,int mu);
void su3_find_heatbath(su3 out,su3 in,su3 staple,double beta,int nhb_steps,rnd_gen *gen);
void su3_find_overrelaxed(su3 out,su3 in,su3 staple,int nov_steps);
void su3_hermitian_prod_double(su3 a,su3 b,double r);
void su3_overrelax(su3 out,su3 in,double w);
void su3_print(su3 U);
void su3_prod_double(su3 a,su3 b,double r);
void su3_prod_with_idouble(su3 a,su3 b,double r);
void su3_put_to_diag(su3 m,color in);
void su3_put_to_id(su3 m);
void su3_put_to_rnd(su3 u_ran,rnd_gen &rnd);
void su3_put_to_zero(su3 m);
void su3_subt(su3 a,su3 b,su3 c);
void su3_subt_complex(su3 a,su3 b,complex c);
void su3_subt_the_prod_color(color a,su3 b,color c);
void su3_subt_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void su3_subt_the_prod_su3_dag(su3 a,su3 b,su3 c);
void su3_summ(su3 a,su3 b,su3 c);
void su3_summassign(su3 a,su3 b);
void su3_summ_real(su3 a,su3 b,double c);
void su3_summ_the_prod_color(color a,su3 b,color c);
void su3_summ_the_prod_double(su3 a,su3 b,double r);
void su3_summ_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void su3_summ_the_trace(complex tr,su3 m);
void su3_trace(complex tr,su3 m);
void su3_traceless_anti_hermitian_part(su3 out,su3 in);
void su3_unitarize_explicitly_inverting(su3 new_link,su3 prop_link);
void su3_unitarize_maximal_trace_projecting(su3 U,su3 M);
void su3_unitarize_maximal_trace_projecting_iteration(su3 U,su3 M);
void su3_unitarize_orthonormalizing(su3 o,su3 i);
void su3spinspin_put_to_zero(su3spinspin m);
void unsafe_color_prod_complex(color out,color in,complex factor);
void unsafe_color_prod_complex_conj(color out,color in,complex factor);
void unsafe_color_prod_su3(color a,color b,su3 c);
void unsafe_color_prod_su3_dag(color a,color b,su3 c);
void unsafe_dirac_prod_spincolor(spincolor out,dirac_matr *m,spincolor in);
void unsafe_spincolor_prod_complex(spincolor out,spincolor in,complex factor);
void unsafe_spincolor_prod_dirac(spincolor out,spincolor in,dirac_matr *m);
void unsafe_spincolor_summ_with_ifactor(spincolor out,spincolor a,spincolor b,double factor);
void unsafe_spincolor_summassign_the_prod_idouble(spincolor out,spincolor in,double factor);
void spincolor_prodassign_idouble(spincolor out,double factor);
void safe_su3_dag_summ_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void safe_su3_dag_subt_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void unsafe_su3_dag_dirac_prod_spincolor(spincolor out,su3 U,dirac_matr *m,spincolor in);
void unsafe_su3_dag_dirac_summ_the_prod_spincolor(spincolor out,su3 U,dirac_matr *m,spincolor in);
void unsafe_su3_dag_prod_color(color a,su3 b,color c);
void unsafe_su3_dag_prod_spincolor(spincolor out,su3 U,spincolor in);
void unsafe_su3_dag_prod_su3(su3 a,su3 b,su3 c);
void unsafe_su3_dag_prod_su3_dag(su3 a,su3 b,su3 c);
void unsafe_su3_dag_subt_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void unsafe_su3_dag_summ_the_prod_spincolor(spincolor out,su3 U,spincolor in);
void unsafe_su3_dirac_prod_spincolor(spincolor out,su3 U,dirac_matr *m,spincolor in);
void unsafe_su3_dirac_subt_the_prod_spincolor(spincolor out,su3 U,dirac_matr *m,spincolor in);
void unsafe_su3_explicit_inverse(su3 invU,su3 U);
void unsafe_su3_hermitian(su3 out,su3 in);
void unsafe_su3_prod_color(color a,su3 b,color c);
void unsafe_su3_prod_complex(su3 a,su3 b,complex c);
void unsafe_su3_prod_conj_complex(su3 a,su3 b,complex c);
void unsafe_su3_prod_spincolor(spincolor out,su3 U,spincolor in);
void unsafe_su3_prod_su3(su3 a,su3 b,su3 c);
void unsafe_su3_prod_su3_dag(su3 a,su3 b,su3 c);
void unsafe_su3_taylor_exponentiate(su3 out,su3 in,int order);
void unsafe_su3spinspin_prod_complex(su3spinspin out,su3spinspin in,complex factor);
#endif
