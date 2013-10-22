#ifndef _SPIN_H
#define _SPIN_H

namespace nissa
{
  double real_part_of_trace_spinspin_prod_spinspin_dag(spinspin a,spinspin b);
  void as2t_saturate(complex out,as2t a,as2t b);
  void get_color_from_colorspinspin(color out,colorspinspin in,int id1,int id2);
  void get_color_from_spincolor(color out,spincolor in,int id);
  void get_spincolor_from_colorspinspin(spincolor out,colorspinspin in,int id_source);
  void get_spincolor_from_su3spinspin(spincolor out,su3spinspin in,int id_source,int ic_source);
  void put_color_into_colorspinspin(colorspinspin out,color in,int id1,int id2);
  void put_color_into_spincolor(spincolor out,color in,int id);
  void put_spincolor_into_colorspinspin(colorspinspin out,spincolor in,int id_source);
  void put_spincolor_into_su3spinspin(su3spinspin out,spincolor in,int id_source,int ic_source);
  void safe_dirac_prod_spin(spin out,dirac_matr *m,spin in);
  void safe_spin_prod_spinspin(spin out,spin a,spinspin b);
  void safe_spinspin_complex_prod(spinspin a,spinspin b,complex c);
  void safe_spinspin_hermitian(spinspin b,spinspin a);
  void safe_spinspin_prod_idouble(spinspin a,spinspin b,double c);
  void safe_spinspin_prod_spin(spin out,spinspin a,spin b);
  void safe_spinspin_prod_spinspin(spinspin out,spinspin a,spinspin b);
  void safe_spinspin_prod_spinspin_dag(spinspin out,spinspin a,spinspin b);
  void spin_copy(spin b,spin a);
  void spin_print(spin s);
  void spin_prod_double(spin a,spin b,double c);
  void spin_prodassign_double(spin a,double b);
  void spin_put_to_zero(spin a);
  void spin_subt(spin a,spin b,spin c);
  void spin_subt_the_complex_conj2_prod(spin a,spin b,complex c);
  void spin_subt_the_complex_prod(spin a,spin b,complex c);
  void spin_subtassign(spin a,spin b);
  void spin_summ(spin a,spin b,spin c);
  void spin_summ_the_complex_conj2_prod(spin a,spin b,complex c);
  void spin_summ_the_complex_prod(spin a,spin b,complex c);
  void spin_summassign(spin a,spin b);
  void spinspin_copy(spinspin b,spinspin a);
  void spinspin_print(spinspin s);
  void spinspin_prod_double(spinspin a,spinspin b,double c);
  void spinspin_prodassign_double(spinspin a,double b);
  void spinspin_prodassign_idouble(spinspin a,double b);
  void spinspin_put_to_id(spinspin a);
  void spinspin_put_to_zero(spinspin a);
  void spinspin_subt(spinspin a,spinspin b,spinspin c);
  void spinspin_subt_the_complex_conj2_prod(spinspin a,spinspin b,complex c);
  void spinspin_subt_the_complex_prod(spinspin a,spinspin b,complex c);
  void spinspin_subtassign(spinspin a,spinspin b);
  void spinspin_summ(spinspin a,spinspin b,spinspin c);
  void spinspin_summ_the_complex_conj2_prod(spinspin a,spinspin b,complex c);
  void spinspin_summ_the_complex_prod(spinspin a,spinspin b,complex c);
  void spinspin_summ_the_spinspin_dag_prod(spinspin out,spinspin a,spinspin b);
  void spinspin_summ_the_spinspin_prod(spinspin out,spinspin a,spinspin b);
  void spinspin_summassign(spinspin a,spinspin b);
  void summ_the_trace_dirac_prod_spinspin(complex c,dirac_matr *a,spinspin b);
  void summ_the_trace_spinspin(complex c,spinspin a);
  void trace_dirac_prod_spinspin(complex c,dirac_matr *a,spinspin b);
  void trace_spinspin(complex c,spinspin a);
  void unsafe_dirac_prod_spin(spin out,dirac_matr *m,spin in);
  void unsafe_spin_prod_spinspin(spin out,spin a,spinspin b);
  void unsafe_spinspin_complex_prod(spinspin a,spinspin b,complex c);
  void unsafe_spinspin_hermitian(spinspin b,spinspin a);
  void unsafe_spinspin_prod_idouble(spinspin a,spinspin b,double c);
  void unsafe_spinspin_prod_spin(spin out,spinspin a,spin b);
  void unsafe_spinspin_prod_spinspin(spinspin out,spinspin a,spinspin b);
  void unsafe_spinspin_prod_spinspin_dag(spinspin out,spinspin a,spinspin b);
  void rotate_spinspin_to_physical_basis(spinspin s,int rsi,int rso);
}

#endif