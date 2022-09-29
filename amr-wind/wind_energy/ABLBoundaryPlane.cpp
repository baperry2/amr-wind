#include "amr-wind/CFDSim.H"
#include "amr-wind/wind_energy/ABLBoundaryPlane.H"
#include "amr-wind/wind_energy/ABLFillInflow.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/utilities/ncutils/nc_interface.H"
#include <AMReX_PlotFileUtil.H>

namespace amr_wind {

namespace {

//! Return closest index (from lower) of value in vector
AMREX_FORCE_INLINE int
closest_index(const amrex::Vector<amrex::Real>& vec, const amrex::Real value)
{
    auto const it = std::upper_bound(vec.begin(), vec.end(), value);
    AMREX_ALWAYS_ASSERT(it != vec.end());

    const int idx = std::distance(vec.begin(), it);
    return std::max(idx - 1, 0);
}

//! Return indices perpendicular to normal
template <typename T = amrex::GpuArray<int, 2>>
AMREX_FORCE_INLINE T perpendicular_idx(const int normal)
{
    switch (normal) {
    case 0:
        return T{1, 2};
    case 1:
        return T{0, 2};
    case 2:
        return T{0, 1};
    default:
        amrex::Abort("Invalid normal value to determine perpendicular indices");
    }
    return T{-1, -1};
}

//! Return offset vector
AMREX_FORCE_INLINE amrex::IntVect offset(const int face_dir, const int normal)
{
    amrex::IntVect offset(amrex::IntVect::TheDimensionVector(normal));
    if (face_dir == 1) {
        for (auto& o : offset) {
            o *= -1;
        }
    }
    return offset;
}

#ifdef AMR_WIND_USE_NETCDF
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
plane_idx(const int i, const int j, const int k, const int perp, const int lo)
{
    return (static_cast<int>(perp == 0) * i + static_cast<int>(perp == 1) * j +
            static_cast<int>(perp == 2) * k) -
           lo;
}

AMREX_FORCE_INLINE std::string level_name(int lev)
{
    return "level_" + std::to_string(lev);
}
#endif

} // namespace

void InletData::resize(const int size)
{
    m_data_n.resize(size);
    m_data_np1.resize(size);
    m_data_interp.resize(size);
}

void InletData::define_plane(const amrex::Orientation ori)
{
    m_data_n[ori] = std::make_unique<PlaneVector>();
    m_data_np1[ori] = std::make_unique<PlaneVector>();
    m_data_interp[ori] = std::make_unique<PlaneVector>();
}

void InletData::define_level_data(
    const amrex::Orientation ori, const amrex::Box& bx, const size_t nc)
{
    if (!this->is_populated(ori)) {
        return;
    }
    m_data_n[ori]->push_back(amrex::FArrayBox(bx, nc));
    m_data_np1[ori]->push_back(amrex::FArrayBox(bx, nc));
    m_data_interp[ori]->push_back(amrex::FArrayBox(bx, nc));
}

void InletData::read_data_native(
    const amrex::OrientationIter oit,
    amrex::BndryRegister& bndry_n,
    amrex::BndryRegister& bndry_np1,
    const int lev,
    const Field* fld,
    const amrex::Real time,
    const amrex::Vector<amrex::Real>& times)
{
    const size_t nc = fld->num_comp();
    const int nstart = m_components[fld->id()];

    const int idx = closest_index(times, time);
    const int idxp1 = idx + 1;

    m_tn = times[idx];
    m_tnp1 = times[idxp1];

    auto ori = oit();

    AMREX_ALWAYS_ASSERT(((m_tn <= time) && (time <= m_tnp1)));
    AMREX_ALWAYS_ASSERT(fld->num_comp() == bndry_n[ori].nComp());
    AMREX_ASSERT(bndry_n[ori].boxArray() == bndry_np1[ori].boxArray());

    const int normal = ori.coordDir();
    const auto& bbx = (*m_data_n[ori])[lev].box();
    const amrex::IntVect v_offset = offset(ori.faceDir(), normal);

    amrex::MultiFab bndry(
        bndry_n[ori].boxArray(), bndry_n[ori].DistributionMap(),
        bndry_n[ori].nComp(), 0, amrex::MFInfo());

    for (amrex::MFIter mfi(bndry); mfi.isValid(); ++mfi) {

        const auto& vbx = mfi.validbox();
        const auto& bndry_n_arr = bndry_n[ori].array(mfi);
        const auto& bndry_arr = bndry.array(mfi);

        const auto& bx = bbx & vbx;
        if (bx.isEmpty()) {
            continue;
        }

        amrex::ParallelFor(
            bx, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                      /* std::cout << "idx " << i << " " << j << " " << k << " " << n << " "
                                << bndry_n_arr(i, j, k, n) << " "
                                << bndry_n_arr(i + v_offset[0], j + v_offset[1], k + v_offset[2], n) << std::endl; */
                bndry_arr(i, j, k, n) =
                    0.5 *
                    (bndry_n_arr(i, j, k, n) +
                     bndry_n_arr(
                         i + v_offset[0], j + v_offset[1], k + v_offset[2], n));
            });
    }

    bndry.copyTo((*m_data_n[ori])[lev], 0, nstart, nc);

    for (amrex::MFIter mfi(bndry); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();
        const auto& bndry_np1_arr = bndry_np1[ori].array(mfi);
        const auto& bndry_arr = bndry.array(mfi);

        const auto& bx = bbx & vbx;
        if (bx.isEmpty()) {
            continue;
        }

        amrex::ParallelFor(
            bx, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                bndry_arr(i, j, k, n) =
                    0.5 *
                    (bndry_np1_arr(i, j, k, n) +
                     bndry_np1_arr(
                         i + v_offset[0], j + v_offset[1], k + v_offset[2], n));
            });
    }

    bndry.copyTo((*m_data_np1[ori])[lev], 0, nstart, nc);
}

void InletData::interpolate(const amrex::Real time)
{
    m_tinterp = time;
    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if (!this->is_populated(ori)) {
            continue;
        }

        const int lnlevels = m_data_n[ori]->size();
        for (int lev = 0; lev < lnlevels; ++lev) {

            const auto& datn = (*m_data_n[ori])[lev];
            const auto& datnp1 = (*m_data_np1[ori])[lev];
            auto& dati = (*m_data_interp[ori])[lev];
            amrex::Print() << ori << " " << lev << " INTERP " <<  m_tnp1 << " "  << m_tn << " " << m_tinterp << std::endl
                      << "VAR 1: " << datn.min( datn.box(),0) << " " << datn.max( datn.box(),0) << " "
                      << datnp1.min( datn.box(),0) << " " << datnp1.max( datn.box(),0) << " "
                      << std::endl
                      << "VAR 2: " << datn.min( datn.box(),1) << " " << datn.max( datn.box(),1) << " "
                      << datnp1.min( datn.box(),1) << " " << datnp1.max( datn.box(),1) << " "
                      << std::endl
                      << "VAR 3: " << datn.min( datn.box(),2) << " " << datn.max( datn.box(),2) << " "
                      << datnp1.min( datn.box(),2) << " " << datnp1.max( datn.box(),2) << " "
                      << std::endl
                      << "VAR 4: " << datn.min( datn.box(),3) << " " << datn.max( datn.box(),3) << " "
                      << datnp1.min( datn.box(),3) << " " << datnp1.max( datn.box(),3) << " "
                      << std::endl;
            dati.linInterp<amrex::RunOn::Device>(
                datn, 0, datnp1, 0, m_tn, m_tnp1, m_tinterp, datn.box(), 0,
                dati.nComp());
        }
    }
}

bool InletData::is_populated(amrex::Orientation ori) const
{
    return m_data_n[ori] != nullptr;
}

ABLBoundaryPlane::ABLBoundaryPlane(CFDSim& sim)
  : m_time(sim.time()), m_repo(sim.repo()), m_mesh(sim.mesh())
{
    m_mbc = sim.mbc();
    //sim.run_mbc();
    amrex::ParmParse pp("ABL");
    int pp_io_mode = -1;
    pp.query("bndry_io_mode", pp_io_mode);
    switch (pp_io_mode) {
    case 0:
        m_io_mode = io_mode::output;
        m_is_initialized = true;
        break;
    case 1:
        m_io_mode = io_mode::input;
        m_is_initialized = true;
        break;
    default:
        m_io_mode = io_mode::undefined;
        m_is_initialized = false;
        return;
    }

    pp.query("bndry_write_frequency", m_write_frequency);
    pp.queryarr("bndry_planes", m_planes);
    pp.query("bndry_output_start_time", m_out_start_time);
    pp.queryarr("bndry_var_names", m_var_names);
    pp.get("bndry_file", m_filename);
    pp.query("bndry_output_format", m_out_fmt);

#ifndef AMR_WIND_USE_NETCDF
    if (m_out_fmt == "netcdf") {
        amrex::Print()
            << "Warning: boundary output format using netcdf must link netcdf "
               "library, changing output to native format"
            << std::endl;
        m_out_fmt = "native";
    }
#endif

    if (!(m_out_fmt == "native" || m_out_fmt == "netcdf" || m_out_fmt == "erf-multiblock")) {
        amrex::Print() << "Warning: boundary output format not recognized, "
                          "changing to native format"
                       << std::endl;
        m_out_fmt = "native";
    }

    // only used for native format
    m_time_file = m_filename + "/time.dat";
}

void ABLBoundaryPlane::post_init_actions()
{
    if (!m_is_initialized) {
        return;
    }
    initialize_data();
    write_header();
    write_file();
    read_header();
    read_file();
}

void ABLBoundaryPlane::pre_advance_work()
{
    if (!m_is_initialized) {
        return;
    }
    read_file();
}

void ABLBoundaryPlane::post_advance_work()
{
    if (!m_is_initialized) {
        return;
    }
    write_file();
}

void ABLBoundaryPlane::initialize_data()
{
    BL_PROFILE("amr-wind::ABLBoundaryPlane::initialize_data");
    for (const auto& fname : m_var_names) {
        if (m_repo.field_exists(fname)) {
            auto& fld = m_repo.get_field(fname);
            if (m_io_mode == io_mode::input) {
                fld.register_fill_patch_op<ABLFillInflow>(
                    m_mesh, m_time, *this);
            }
            m_fields.emplace_back(&fld);
        } else {
            amrex::Abort(
                "ABLBoundaryPlane: invalid variable requested: " + fname);
        }
    }
    if (m_io_mode == io_mode::output and m_out_fmt == "erf-multiblock") {
      amrex::Abort("ABLBoundaryPlane: can't output data in erf-multiblock mode");
    }
}

void ABLBoundaryPlane::write_header()
{
    BL_PROFILE("amr-wind::ABLBoundaryPlane::write_header");
    return;
}

void ABLBoundaryPlane::write_file()
{
    BL_PROFILE("amr-wind::ABLBoundaryPlane::write_file");
    return;
}

void ABLBoundaryPlane::read_header()
{
    BL_PROFILE("amr-wind::ABLBoundaryPlane::read_header");
    if (m_io_mode != io_mode::input) {
        return;
    }

    // FIXME: overallocate this for now
    m_in_data.resize(2 * AMREX_SPACEDIM);

    if (m_out_fmt == "native") {

        int time_file_length = 0;

        if (amrex::ParallelDescriptor::IOProcessor()) {

            std::string line;
            std::ifstream time_file(m_time_file);
            if (!time_file.good()) {
                amrex::Abort("Cannot find time file: " + m_time_file);
            }
            while (std::getline(time_file, line)) {
                ++time_file_length;
            }

            time_file.close();
        }

        amrex::ParallelDescriptor::Bcast(
            &time_file_length, 1,
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());

        m_in_times.resize(time_file_length);
        m_in_timesteps.resize(time_file_length);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ifstream time_file(m_time_file);
            for (int i = 0; i < time_file_length; ++i) {
                time_file >> m_in_timesteps[i] >> m_in_times[i];
            }
            time_file.close();
        }

        amrex::ParallelDescriptor::Bcast(
            m_in_timesteps.data(), time_file_length,
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());

        amrex::ParallelDescriptor::Bcast(
            m_in_times.data(), time_file_length,
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());

        int nc = 0;
        for (auto* fld : m_fields) {
            m_in_data.component(static_cast<int>(fld->id())) = nc;
            nc += fld->num_comp();
        }

        // FIXME: need to generalize to lev > 0 somehow
        const int lev = 0;
        for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();

            // FIXME: would be safer and less storage to not allocate all of
            // these but we do not use m_planes for input and need to detect
            // mass inflow from field bcs same for define level data below
            m_in_data.define_plane(ori);

            const amrex::Box& minBox = m_mesh.boxArray(lev).minimalBox();

            amrex::IntVect plo(minBox.loVect());
            amrex::IntVect phi(minBox.hiVect());
            const int normal = ori.coordDir();
            plo[normal] = ori.isHigh() ? minBox.hiVect()[normal] + 1 : -1;
            phi[normal] = ori.isHigh() ? minBox.hiVect()[normal] + 1 : -1;
            const amrex::Box pbx(plo, phi);
            m_in_data.define_level_data(ori, pbx, nc);
        }
    } else if (m_out_fmt == "erf-multiblock") {

        m_in_times.push_back(-1.0e13); // create space for storing time at erf old and new timestep
        m_in_times.push_back(-1.0e13);
        int nc = 0;
        for (auto* fld : m_fields) {
            m_in_data.component(static_cast<int>(fld->id())) = nc;
            nc += fld->num_comp();
        }

        // FIXME: need to generalize to lev > 0 somehow
        const int lev = 0;
        for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();

            // FIXME: would be safer and less storage to not allocate all of
            // these but we do not use m_planes for input and need to detect
            // mass inflow from field bcs same for define level data below
            m_in_data.define_plane(ori);

            const amrex::Box& minBox = m_mesh.boxArray(lev).minimalBox();

            amrex::IntVect plo(minBox.loVect());
            amrex::IntVect phi(minBox.hiVect());
            const int normal = ori.coordDir();
            plo[normal] = ori.isHigh() ? minBox.hiVect()[normal] + 1 : -1;
            phi[normal] = ori.isHigh() ? minBox.hiVect()[normal] + 1 : -1;
            const amrex::Box pbx(plo, phi);
            m_in_data.define_level_data(ori, pbx, nc);
        }
    }
}

void ABLBoundaryPlane::read_erf()
{
    amrex::Real time = m_time.new_time();
    AMREX_ALWAYS_ASSERT(m_in_times[0] <= time); // Can't go back in time for ERF data
    // return early if current erf data can still be interpolated in time
    if ((m_in_data.tn() <= time) && (time < m_in_data.tnp1())) {
        m_in_data.interpolate(time);
        return;
    }

    // Get current ERF time values
    mbc()->PopulateErfTimesteps(m_in_times.data());
    m_in_times[1] += 1e-12; // lets case where time = m_in_times[1] be valid
    AMREX_ALWAYS_ASSERT((m_in_times[0]<= time) && (time <= m_in_times[1]));
    const int index = 0;
    const int lev = 0;
    // FIXME FIXME TODO DELETE
    mbc()->SetBoxLists();

    for (auto* fld : m_fields) {

      auto& field = *fld;
      const auto& geom = field.repo().mesh().Geom();

      amrex::Box domain = geom[lev].Domain();
      amrex::BoxArray ba(domain);
      amrex::DistributionMapping dm{ba};

      std::cout << " BOX ARRAY " << ba << std::endl << " DM " << dm << std::endl ;

      amrex::BndryRegister bndry1(ba, dm, m_in_rad, m_out_rad, m_extent_rad, field.num_comp());
      amrex::BndryRegister bndry2(ba, dm, m_in_rad, m_out_rad, m_extent_rad, field.num_comp());
      
      if (field.name() == "velocity") {
        bndry1.setVal(1.0e13); // 1.0e13
        bndry2.setVal(1.0e13); // 1.0e13
      } else if (field.name() == "temperature") {
        bndry1.setVal(1.0e13); // 1.0e13
        bndry2.setVal(1.0e13); // 1.0e13
      }

      for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if ((!m_in_data.is_populated(ori)) ||
            (field.bc_type()[ori] != BC::mass_inflow)) {
          //continue;
        }
        if (field.bc_type()[ori] == BC::mass_inflow and time > 0.0) {
          if ( field.name() == "temperature") {
            mbc()->CopyToBoundaryRegister(bndry1, bndry2, ori);
          } else if ( field.name() == "velocity") {
            // mbc()->CopyToBoundaryRegister(bndry1, bndry2, ori);
          }
        } else {
          if (field.name() == "temperature") {
            bndry1[ori].setVal(300.0);
            bndry2[ori].setVal(300.0);
          } else if (field.name() == "velocity") {
            bndry1[ori].setVal(10.0, 0, 1);
            bndry2[ori].setVal(10.0, 0, 1);
            bndry1[ori].setVal( 0.0, 1, 1);
            bndry2[ori].setVal( 0.0, 1, 1);
            bndry1[ori].setVal( 0.0, 2, 1);
            bndry2[ori].setVal( 0.0, 2, 1);
          }
        }
        /*
        amrex::IntVect nghost(0);
        amrex::NonLocalBC::MultiBlockCommMetaData *cmd_full_tmp =
          new amrex::NonLocalBC::MultiBlockCommMetaData(bndry1[ori].multiFab(), domain,
                                                        bndry2[ori].multiFab(), nghost, mbc()->dtos_etoa);
        */

        //std::cout << ori << " after break " << std::endl;

        //std::cout << "BNDRY REG " << field.name() << " " << ori << " " << std::endl;
        /*
          std::string facename1 =
          amrex::Concatenate(filename1 + '_', ori, 1);
          std::string facename2 =
          amrex::Concatenate(filename2 + '_', ori, 1);

          bndry1[ori].read(facename1);
          bndry2[ori].read(facename2);
        */
        m_in_data.read_data_native(oit, bndry1, bndry2, lev, fld, time, m_in_times);

      }
    }
    m_in_data.interpolate(time);
}


void ABLBoundaryPlane::read_file()
{
    BL_PROFILE("amr-wind::ABLBoundaryPlane::read_file");
    if (m_io_mode != io_mode::input) {
        return;
    }

    if (m_out_fmt == "erf-multiblock") {
      read_erf();
      return;
    }

    // populate planes and interpolate
    amrex::Real time = m_time.new_time();
    if (m_in_times[0] > time) {
      std::cout << "resetting time to " <<  m_in_times[0] + 1e-6 << std::endl;
      time = m_in_times[0] + 1e-6;
    }
    AMREX_ALWAYS_ASSERT((m_in_times[0] <= time) && (time < m_in_times.back()));

    // return early if current data files can still be interpolated in time
    if ((m_in_data.tn() <= time) && (time < m_in_data.tnp1())) {
        m_in_data.interpolate(time);
        return;
    }

    if (m_out_fmt == "native") {

        const int index = closest_index(m_in_times, time);
        const int t_step1 = m_in_timesteps[index];
        const int t_step2 = m_in_timesteps[index + 1];

        AMREX_ALWAYS_ASSERT(
            (m_in_times[index] <= time) && (time <= m_in_times[index + 1]));

        const std::string chkname1 =
            m_filename + amrex::Concatenate("/bndry_output", t_step1);
        const std::string chkname2 =
            m_filename + amrex::Concatenate("/bndry_output", t_step2);

        const std::string level_prefix = "Level_";

        const int lev = 0;
        for (auto* fld : m_fields) {

            auto& field = *fld;
            const auto& geom = field.repo().mesh().Geom();

            amrex::Box domain = geom[lev].Domain();
            amrex::BoxArray ba(domain);
            amrex::DistributionMapping dm{ba};

            amrex::BndryRegister bndry1(
                ba, dm, m_in_rad, m_out_rad, m_extent_rad, field.num_comp());
            amrex::BndryRegister bndry2(
                ba, dm, m_in_rad, m_out_rad, m_extent_rad, field.num_comp());

            bndry1.setVal(1.0e13);
            bndry2.setVal(1.0e13);

            std::string filename1 = amrex::MultiFabFileFullPrefix(
                lev, chkname1, level_prefix, field.name());
            std::string filename2 = amrex::MultiFabFileFullPrefix(
                lev, chkname2, level_prefix, field.name());

            for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
                auto ori = oit();

                if ((!m_in_data.is_populated(ori)) ||
                    (field.bc_type()[ori] != BC::mass_inflow)) {
                  //std::cout << ori << " " << m_in_data.is_populated(ori) << " " << (field.bc_type()[ori]== BC::mass_inflow) << " before break " << std::endl;
                    continue;
                }
                //std::cout << ori << " afetrr break " << std::endl;

                std::string facename1 =
                    amrex::Concatenate(filename1 + '_', ori, 1);
                std::string facename2 =
                    amrex::Concatenate(filename2 + '_', ori, 1);

                bndry1[ori].read(facename1);
                bndry2[ori].read(facename2);

                m_in_data.read_data_native(
                    oit, bndry1, bndry2, lev, fld, time, m_in_times);
            }
        }
    }

    m_in_data.interpolate(time);
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void ABLBoundaryPlane::populate_data(
    const int lev,
    const amrex::Real time,
    Field& fld,
    amrex::MultiFab& mfab) const
{

    BL_PROFILE("amr-wind::ABLBoundaryPlane::populate_data");

    if (m_io_mode != io_mode::input) {
        return;
    }

    AMREX_ALWAYS_ASSERT(
        ((m_in_data.tn() <= time) || (time <= m_in_data.tnp1())));
    AMREX_ALWAYS_ASSERT(amrex::Math::abs(time - m_in_data.tinterp()) < 1e-12);

    for (amrex::OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if ((!m_in_data.is_populated(ori)) ||
            (fld.bc_type()[ori] != BC::mass_inflow)) {
            continue;
        }
        // std::cout << "POPULATING DATRA FOR ORIENT " << ori << " AT TIME " << time << std::endl;

        // Ensure the fine level does not touch the inflow boundary
        if (lev > 0) {
            const amrex::Box& minBox = m_mesh.boxArray(lev).minimalBox();
            if (box_intersects_boundary(minBox, lev, ori)) {
                amrex::Abort(
                    "Fine level intersects inflow boundary, not supported "
                    "yet.");
            } else {
                continue;
            }
        }

        // Ensure inflow data exists at this level
        if (lev >= m_in_data.nlevels(ori)) {
            amrex::Abort("No inflow data at this level.");
        }

        // const int normal = ori.coordDir();
        // const amrex::GpuArray<int, 2> perp = perpendicular_idx(normal);

        const size_t nc = mfab.nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mfab, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {

            const auto& sbx = mfi.growntilebox(1);
            const auto& src = m_in_data.interpolate_data(ori, lev);
            const auto& bx = sbx & src.box();
            if (bx.isEmpty()) {
                continue;
            }

            const auto& dest = mfab.array(mfi);
            const auto& src_arr = src.array();
            const int nstart = m_in_data.component(static_cast<int>(fld.id()));
            amrex::ParallelFor(
                bx, nc,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    dest(i, j, k, n) = src_arr(i, j, k, n + nstart);
                });
        }
    }

    const auto& geom = fld.repo().mesh().Geom();
    mfab.EnforcePeriodicity(
        0, mfab.nComp(), amrex::IntVect(1), geom[lev].periodicity());
}

//! True if box intersects the boundary
bool ABLBoundaryPlane::box_intersects_boundary(
    const amrex::Box& bx, const int lev, const amrex::Orientation ori) const
{
    const amrex::Box& domBox = m_mesh.Geom(lev).Domain();
    const int normal = ori.coordDir();
    amrex::IntVect plo(domBox.loVect());
    amrex::IntVect phi(domBox.hiVect());
    plo[normal] = ori.isHigh() ? domBox.loVect()[normal] : 0;
    phi[normal] = ori.isHigh() ? domBox.hiVect()[normal] : 0;
    const amrex::Box pbx(plo, phi);
    const auto& intersection = bx & pbx;
    return !intersection.isEmpty();
}

} // namespace amr_wind
