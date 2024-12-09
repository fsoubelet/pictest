#ifndef XTRACK_PICTRACK_H
#define XTRACK_PICTRACK_H

#include <math.h>

/*gpufun*/
void PICTRACK_attribute_cells(PICTRACKData el, MeshGridPropertiesData mesh, LocalParticle* part0){
    int const nx = MeshGridPropertiesData_get_nx(mesh);
    int const ny = MeshGridPropertiesData_get_ny(mesh);
    int const nz = MeshGridPropertiesData_get_nz(mesh);

    double const minx = MeshGridPropertiesData_get_minx(mesh);
    double const miny = MeshGridPropertiesData_get_miny(mesh);
    double const minz = MeshGridPropertiesData_get_minz(mesh);

    double const delta_x = MeshGridPropertiesData_get_delta_x(mesh);
    double const delta_y = MeshGridPropertiesData_get_delta_y(mesh);
    double const delta_z = MeshGridPropertiesData_get_delta_z(mesh);

    //start_per_particle_block (part0->part)

        // I WANT TO GET A VALUE FOR THE IDX OF THE PARTICLE
        int const idx     = LocalParticle_get_idx(part0);  // TODO: find a way?
        double const x    = LocalParticle_get_x(part0);
        double const y    = LocalParticle_get_y(part0);
        double const zeta = LocalParticle_get_zeta(part0);

        int const cell = floor((x - minx) / delta_x) * nx  // integer in horizontal
                       + floor((y - miny) / delta_y) * ny  // integer in vertical
                       + floor((zeta - minz) / delta_z);   // integer in longitudinal

        // Set that cell attribution value for this particle idx
        PICTRACKData_set__attributions(mesh, idx, cell);

    //end_per_particle_block
}

#endif