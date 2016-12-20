#include "mjd_siggen.h"
#include "calc_signal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int read_velocity_table(struct velocity_lookup* v_lookup, MJD_Siggen_Setup *setup){

  int vlook_sz = 0;
  char  line[MAX_LINE], *c;
  FILE  *fp;
  int   v_lookup_len;
  struct velocity_lookup *tmp;
  
  double be=1.3e7, bh=1.2e7, thetae=200.0, thetah=200.0;  // parameters for temperature correction
  double pwre=-1.680, pwrh=-2.398, mue=5.66e7, muh=1.63e9; //     adopted for Ge   DCR Feb 2015

  if (vlook_sz == 0) {
    vlook_sz = 21;
//    if ((v_lookup = (struct velocity_lookup *)
//	 malloc(vlook_sz*sizeof(*v_lookup))) == NULL) {
//      error("malloc failed in setup_velo\n");
//      return -1;
//    }
  }
  if ((fp = fopen(setup->drift_name, "r")) == NULL){
    error("failed to open velocity lookup table file: '%s'\n", setup->drift_name);
    return -1;
  }
  line[0] = '#';
  c = line;
  while ((line[0] == '#' || line[0] == '\0') && c != NULL) c = fgets(line, MAX_LINE, fp);
  if (c == NULL) {
    error("Failed to read velocity lookup table from file: %s\n", setup->drift_name);
    fclose(fp);
    return -1;
  }
  TELL_CHATTY("Drift velocity table:\n"
	      "  e          e100    e110    e111    h100    h110    h111\n");   
  for (v_lookup_len = 0; ;v_lookup_len++){
    if (v_lookup_len == vlook_sz - 1){
      vlook_sz += 10;
      if ((tmp = (struct velocity_lookup *)
	   realloc(v_lookup, vlook_sz*sizeof(*v_lookup))) == NULL){
	error("realloc failed in setup_velo\n");
	fclose(fp);
	return -1;
      }
      v_lookup = tmp;
    }
    if (sscanf(line, "%f %f %f %f %f %f %f", 
	       &v_lookup[v_lookup_len].e,
	       &v_lookup[v_lookup_len].e100,
	       &v_lookup[v_lookup_len].e110,
	       &v_lookup[v_lookup_len].e111,
	       &v_lookup[v_lookup_len].h100,
	       &v_lookup[v_lookup_len].h110,
	       &v_lookup[v_lookup_len].h111) != 7){
      break; //assume EOF
    }	   
    //v_lookup[v_lookup_len].e *= 100; /*V/m*/
    tmp = &v_lookup[v_lookup_len];
    TELL_CHATTY("%10.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n",
		tmp->e, tmp->e100, tmp->e110, tmp->e111, tmp->h100, tmp->h110,tmp->h111);
    line[0] = '#';
    while ((line[0] == '#' || line[0] == '\0' ||
	    line[0] == '\n' || line[0] == '\r') && c != NULL) c = fgets(line, MAX_LINE, fp);
    if (c == NULL) break;
    if (line[0] == 'e' || line[0] == 'h') break; /* no more velocities data;
						    now reading temp correction data */
  }

  /* check for and decode temperature correction parameters */
  while (line[0] == 'e' || line[0] == 'h') {
    if (line[0] == 'e' &&
	sscanf(line+2, "%lf %lf %lf %lf", 
	       &mue, &pwre, &be, &thetae) != 4) break;//asume EOF
    if (line[0] == 'h' &&
	sscanf(line+2, "%lf %lf %lf %lf", 
	       &muh, &pwrh, &bh, &thetah) != 4) break;//asume EOF
    if (line[0] == 'e')
      TELL_CHATTY("electrons: mu_0 = %.2e x T^%.4f  B = %.2e  Theta = %.0f\n",
		  mue, pwre, be, thetae);
    if (line[0] == 'h')
      TELL_CHATTY("    holes: mu_0 = %.2e x T^%.4f  B = %.2e  Theta = %.0f\n",
		  muh, pwrh, bh, thetah);

    line[0] = '#';
    while ((line[0] == '#' || line[0] == '\0') && c != NULL) c = fgets(line, MAX_LINE, fp);
    if (c == NULL) break;
  }

  if (v_lookup_len == 0){
    error("Failed to read velocity lookup table from file: %s\n", setup->drift_name);
    return -1;
  }  
  v_lookup_len++;
  if (vlook_sz != v_lookup_len){
    if ((tmp = (struct velocity_lookup *) 
	 realloc(v_lookup, v_lookup_len*sizeof(*v_lookup))) == NULL){
      error("realloc failed in setup_velo. This should not happen\n");
      fclose(fp);
      return -1;
    }
    v_lookup = tmp;
    vlook_sz = v_lookup_len;
  }
  TELL_NORMAL("Drift velocity table has %d rows of data\n", v_lookup_len);
  fclose(fp);
  
  setup->v_lookup_len = v_lookup_len;
  
  return 0;
}


int temperature_modify_velocity_table(float e_temp, float h_temp, struct velocity_lookup* v_lookup_saved, struct velocity_lookup* modified_v_lookup, MJD_Siggen_Setup *setup){
 /*
    apply temperature dependence to mobilities;
    see drift_velocities.doc and tempdep.c
    The drift velocity reduces at higher temperature due to the increasing of
    scattering with the lattice vibration. We used a model by M. Ali Omar and
    L. Reggiani (Solid-State Electronics Vol. 30, No. 12 (1987) 1351) to
    calculate the temperature dependence.
  */
  /* electrons */
  

  int   i;
  int vlook_sz = setup->v_lookup_len;
  struct velocity_lookup  v, v0;

  float sumb_e, sumc_e, sumb_h, sumc_h;
  double be=1.3e7, bh=1.2e7, thetae=200.0, thetah=200.0;  // parameters for temperature correction
  double pwre=-1.680, pwrh=-2.398, mue=5.66e7, muh=1.63e9; //     adopted for Ge   DCR Feb 2015
  double mu_0_1, mu_0_2, v_s_1, v_s_2, E_c_1, E_c_2, e, f;
  
  TELL_NORMAL("Adjusting mobilities for temperature, from %.1f to %.1f\n", REF_TEMP, setup->xtal_temp);
  TELL_CHATTY("Index  field  vel_factor\n");
  mu_0_1 = mue * pow(REF_TEMP, pwre);
  v_s_1 = be * sqrt(tanh(0.5 * thetae / REF_TEMP));
  E_c_1 = v_s_1 / mu_0_1;
  mu_0_2 = mue * pow(e_temp, pwre);
  v_s_2 = be * sqrt(tanh(0.5 * thetae / e_temp));
  E_c_2 = v_s_2 / mu_0_2;
  
  
  for (i = 0; i < vlook_sz; i++){
    e = v_lookup_saved[i].e;
    if (e < 1) continue;
    f = (v_s_2 * (e/E_c_2) / sqrt(1.0 + (e/E_c_2) * (e/E_c_2))) /
        (v_s_1 * (e/E_c_1) / sqrt(1.0 + (e/E_c_1) * (e/E_c_1)));
    modified_v_lookup[i].e100 = v_lookup_saved[i].e100*f;
    modified_v_lookup[i].e110 = v_lookup_saved[i].e110*f;
    modified_v_lookup[i].e111 = v_lookup_saved[i].e111*f;
    TELL_CHATTY("%2d %5.0f %f\n", i, e, f);
  }

  /* holes */
  mu_0_1 = muh * pow(REF_TEMP, pwrh);
  v_s_1 = bh * sqrt(tanh(0.5 * thetah / REF_TEMP));
  E_c_1 = v_s_1 / mu_0_1;
  mu_0_2 = muh * pow(h_temp, pwrh);
  v_s_2 = bh * sqrt(tanh(0.5 * thetah / h_temp));
  E_c_2 = v_s_2 / mu_0_2;
  for (i = 0; i < vlook_sz; i++){
    e = v_lookup_saved[i].e;
    if (e < 1) continue;
    f = (v_s_2 * (e/E_c_2) / sqrt(1.0 + (e/E_c_2) * (e/E_c_2))) /
        (v_s_1 * (e/E_c_1) / sqrt(1.0 + (e/E_c_1) * (e/E_c_1)));
    modified_v_lookup[i].h100 = v_lookup_saved[i].h100*f;
    modified_v_lookup[i].h110 = v_lookup_saved[i].h110*f;
    modified_v_lookup[i].h111 = v_lookup_saved[i].h111*f;
    TELL_CHATTY("%2d %5.0f %f\n", i, e, f);
  }
  /* end of temperature correction */

  for (i = 0; i < vlook_sz; i++){
    v = modified_v_lookup[i];
    modified_v_lookup[i].ea =  0.5 * v.e100 -  4 * v.e110 +  4.5 * v.e111;
    modified_v_lookup[i].eb = -2.5 * v.e100 + 16 * v.e110 - 13.5 * v.e111;
    modified_v_lookup[i].ec =  3.0 * v.e100 - 12 * v.e110 +  9.0 * v.e111;
    modified_v_lookup[i].ha =  0.5 * v.h100 -  4 * v.h110 +  4.5 * v.h111;
    modified_v_lookup[i].hb = -2.5 * v.h100 + 16 * v.h110 - 13.5 * v.h111;
    modified_v_lookup[i].hc =  3.0 * v.h100 - 12 * v.h110 +  9.0 * v.h111;
  }
  modified_v_lookup[0].ebp = modified_v_lookup[0].ecp = modified_v_lookup[0].hbp = modified_v_lookup[0].hcp = 0.0;
  sumb_e = sumc_e = sumb_h = sumc_h = 0.0;
  
  for (i = 1; i < vlook_sz; i++){
    v0 = modified_v_lookup[i-1];
    v = modified_v_lookup[i];
    sumb_e += (v.e - v0.e)*(v0.eb+v.eb)/2;
    sumc_e += (v.e - v0.e)*(v0.ec+v.ec)/2;
    sumb_h += (v.e - v0.e)*(v0.hb+v.hb)/2;
    sumc_h += (v.e - v0.e)*(v0.hc+v.hc)/2;
    modified_v_lookup[i].ebp = sumb_e/v.e;
    modified_v_lookup[i].ecp = sumc_e/v.e;
    modified_v_lookup[i].hbp = sumb_h/v.e;
    modified_v_lookup[i].hcp = sumc_h/v.e;
  }
  return 0;

}
