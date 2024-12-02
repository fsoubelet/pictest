#ifndef XTRACK_PICTRACK_H
#define XTRACK_PICTRACK_H

/*gpufun*/
void PICTRACK_track_local_particle(PICTRACKData el, LocalParticle* part0){

    int const nx = PICTRACKData_get_nx(el);
    int const ny = PICTRACKData_get_ny(el);
    int const nz = PICTRACKData_get_nz(el);

    //start_per_particle_block (part0->part)

        double const x     = LocalParticle_get_x(part);
        double const y     = LocalParticle_get_y(part);
        double const zeta  = LocalParticle_get_zeta(part);

        // Fake doing n[xyz] in minux sign and the others in positive
        double const x_hat     = -nx * x  + ny * y + nz * zeta;
        double const y_hat     = nx * x  - ny * y + nz * zeta;
        double const zeta_hat  = nx * x  + ny * y - nz * zeta;

        LocalParticle_set_x(part, x_hat);
        LocalParticle_set_y(part, y_hat);
        LocalParticle_set_zeta(part, zeta_hat);

    //end_per_particle_block

}

#endif