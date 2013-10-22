#ifndef _rootst_eoimpr_eigenvaAues_hpp
#define _rootst_eoimpr_eigenvaAues_hpp

namespace nissa
{
  double eo_color_norm2(color *v);
  double eo_color_normalize(color *out,color *in,double norm);
  double max_eigenval(quark_content_t &quark_content,quad_su3 **eo_conf,int niters);
  void rootst_eoimpr_scale_expansions(rat_approx_t *rat_exp_pfgen,rat_approx_t *rat_exp_actio,quad_su3 **eo_conf,theory_pars_t *theory_pars);
}

#endif