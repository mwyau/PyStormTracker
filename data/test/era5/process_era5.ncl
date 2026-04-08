load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

; General script to process ERA5 data for PyStormTracker tests.
; Usage: ncl 'dir_in="/path/to/data/"' 'dir_out="./output/"' 'date="2025120100"' process_era5.ncl

begin
    ; --- Configuration with Defaults ---
    if (.not. isvar("dir_in")) then
        dir_in = "./"
    end if
    if (.not. isvar("dir_out")) then
        dir_out = "./"
    end if
    if (.not. isvar("date")) then
        date = "2025120100"
    end if
    if (.not. isvar("lmax_trunc")) then
        lmax_trunc = 42
    end if

    ; Helper to clean and save variables
    procedure clean_save(var_in, name, long_name, units, std_name, prefix, suffix, lat, lon, dir_out, date, grid_str)
    local var, out_name
    begin
        var = var_in
        delete_VarAtts(var, -1)
        var@long_name = long_name
        var@units = units
        if (.not. ismissing(std_name)) then
            var@standard_name = std_name
        end if
        var!0 = "latitude"
        var!1 = "longitude"
        var&latitude = lat
        var&longitude = lon
        
        out_name = dir_out + "era5_" + prefix + "_" + date + "_" + grid_str + suffix + ".nc"
        system("rm -f " + out_name)
        f_o = addfile(out_name, "c")
        f_o->$name$ = var
        delete(f_o)
    end

    res_list = (/"0.25", "2.5"/)

    do r=0, dimsizes(res_list)-1
        res = res_list(r)
        grid_str = res + "x" + res
        
        file_msl = dir_in + "era5_msl_2025-2026_djf_" + grid_str + ".nc"
        file_uv = dir_in + "era5_uv850_2025-2026_djf_" + grid_str + ".nc"
        file_vo = dir_in + "era5_vo850_2025-2026_djf_" + grid_str + ".nc"

        ; 1. Load First Frames
        print("Processing " + grid_str + " for date " + date + "...")
        
        f_msl_src = addfile(file_msl, "r")
        msl_raw = f_msl_src->msl(0,:,:)
        
        f_uv_src = addfile(file_uv, "r")
        u_raw = f_uv_src->u(0,0,::-1,:)
        v_raw = f_uv_src->v(0,0,::-1,:)
        
        f_vo_src = addfile(file_vo, "r")
        vo_raw = f_vo_src->vo(0,0,:,:)
        
        ; Define standard coordinates (double)
        lat = todouble(f_msl_src->latitude)
        lon = todouble(f_msl_src->longitude)
        lat@units = "degrees_north"
        lat@long_name = "latitude"
        lat@standard_name = "latitude"
        lat@stored_direction = "decreasing"
        lon@units = "degrees_east"
        lon@long_name = "longitude"
        lon@standard_name = "longitude"

        ; 2. Calculate VODV and SFVP
        ; NOTE: NCL's uv2vrF and uv2dvF use an opposite sign convention 
        ; to ERA5/PST for North-to-South grids. We negate them here to match.
        vo_calc = uv2vrF_Wrap(u_raw, v_raw)
        dv_calc = uv2dvF_Wrap(u_raw, v_raw)
        sfvp    = uv2sfvpF(u_raw, v_raw)
        sf_latrev = sfvp(0,::-1,:)
        vp_latrev = sfvp(1,::-1,:)

        vo_latrev = vo_calc(::-1,:)
        dv_latrev = dv_calc(::-1,:)
        
        ; 3. Spectral Filtering (T42)
        ; Determine nlon for synthesis
        dims = dimsizes(msl_raw)
        nlon = dims(1)
        
        a = shaeC(msl_raw)
        
        ; Low-pass (T0-42)
        a_t0 = a
        a_t0(:, (lmax_trunc+1):, :) = 0.0
        msl_t0 = shseC(a_t0, nlon)
        msl_t0@long_name = "Mean sea level pressure (T0-42)"
        msl_t0@units = "Pa"
        msl_t0@standard_name = "air_pressure_at_mean_sea_level"
        msl_t0!0 = "latitude"
        msl_t0!1 = "longitude"
        msl_t0&latitude = lat
        msl_t0&longitude = lon
        
        ; Band-pass (T5-42)
        a_t5 = a_t0
        a_t5(:, 0:4, :) = 0.0
        msl_t5 = shseC(a_t5, nlon)
        msl_t5@long_name = "Mean sea level pressure (T5-42)"
        msl_t5@units = "Pa"
        msl_t5@standard_name = "air_pressure_at_mean_sea_level"
        msl_t5!0 = "latitude"
        msl_t5!1 = "longitude"
        msl_t5&latitude = lat
        msl_t5&longitude = lon

        ; 4. Save and Standardize Metadata
        
        ; Helper to clean and save variables
        ; Note: procedure is defined outside or redefined carefully. 
        ; In NCL, defining a procedure inside a loop is not allowed if it's already defined.
        ; I'll use the existing clean_save which I will move before the loop.

        ; --- UV850 ---
        u_s = u_raw
        v_s = v_raw
        delete_VarAtts(u_s, -1)
        u_s@long_name = "U component of wind"
        u_s@units = "m s**-1"
        u_s@standard_name = "eastward_wind"
        u_s!0 = "latitude"
        u_s!1 = "longitude"
        u_s&latitude = lat
        u_s&longitude = lon
        
        delete_VarAtts(v_s, -1)
        v_s@long_name = "V component of wind"
        v_s@units = "m s**-1"
        v_s@standard_name = "northward_wind"
        v_s!0 = "latitude"
        v_s!1 = "longitude"
        v_s&latitude = lat
        v_s&longitude = lon
        
        f_uv_out = dir_out + "era5_uv850_" + date + "_" + grid_str + ".nc"
        system("rm -f " + f_uv_out)
        out_uv = addfile(f_uv_out, "c")
        out_uv->u = u_s
        out_uv->v = v_s
        delete(out_uv)

        ; --- VO850 (raw) ---
        clean_save(vo_raw, "vo", "Vorticity (relative)", "s**-1", "atmosphere_relative_vorticity", "vo850", "", lat, lon, dir_out, date, grid_str)

        ; --- VODV (calculated) ---
        vo_c = vo_latrev
        dv_c = dv_latrev
        delete_VarAtts(vo_c, -1)
        vo_c@long_name = "Vorticity (relative)"
        vo_c@units = "s**-1"
        vo_c@standard_name = "atmosphere_relative_vorticity"
        vo_c!0 = "latitude"
        vo_c!1 = "longitude"
        vo_c&latitude = lat
        vo_c&longitude = lon
        
        delete_VarAtts(dv_c, -1)
        dv_c@long_name = "Divergence"
        dv_c@units = "s**-1"
        dv_c@standard_name = "divergence_of_wind"
        dv_c!0 = "latitude"
        dv_c!1 = "longitude"
        dv_c&latitude = lat
        dv_c&longitude = lon
        
        f_vodv_out = dir_out + "era5_vodv850_" + date + "_" + grid_str + "_ncl.nc"
        system("rm -f " + f_vodv_out)
        out_vodv = addfile(f_vodv_out, "c")
        out_vodv->vo = vo_c
        out_vodv->dv = dv_c
        delete(out_vodv)

        ; --- SFVP (calculated) ---
        sf_c = sf_latrev
        vp_c = vp_latrev
        delete_VarAtts(sf_c, -1)
        sf_c@long_name = "Stream function"
        sf_c@units = "m**2 s**-1"
        sf_c@standard_name = "atmosphere_horizontal_streamfunction"
        sf_c!0 = "latitude"
        sf_c!1 = "longitude"
        sf_c&latitude = lat
        sf_c&longitude = lon
        
        delete_VarAtts(vp_c, -1)
        vp_c@long_name = "Velocity potential"
        vp_c@units = "m**2 s**-1"
        vp_c@standard_name = "atmosphere_horizontal_velocity_potential"
        vp_c!0 = "latitude"
        vp_c!1 = "longitude"
        vp_c&latitude = lat
        vp_c&longitude = lon
        
        f_sfvp_out = dir_out + "era5_sfvp850_" + date + "_" + grid_str + "_ncl.nc"
        system("rm -f " + f_sfvp_out)
        out_sfvp = addfile(f_sfvp_out, "c")
        out_sfvp->sf = sf_c
        out_sfvp->vp = vp_c
        delete(out_sfvp)

        ; --- MSL Raw ---
        clean_save(msl_raw, "msl", "Mean sea level pressure", "Pa", "air_pressure_at_mean_sea_level", "msl", "", lat, lon, dir_out, date, grid_str)

        ; --- MSL Filtered ---
        clean_save(msl_t0, "msl", "Mean sea level pressure (T0-42)", "Pa", "air_pressure_at_mean_sea_level", "msl", "_t0-42_ncl", lat, lon, dir_out, date, grid_str)

        clean_save(msl_t5, "msl", "Mean sea level pressure (T5-42)", "Pa", "air_pressure_at_mean_sea_level", "msl", "_t5-42_ncl", lat, lon, dir_out, date, grid_str)

        ; Cleanup for next iteration
        delete([/msl_raw, u_raw, v_raw, vo_raw, lat, lon, vo_calc, dv_calc, vo_latrev, dv_latrev, sfvp, sf_latrev, vp_latrev, dims, a, a_t0, msl_t0, a_t5, msl_t5, u_s, v_s, vo_c, dv_c, sf_c, vp_c/])
        print("Successfully processed " + grid_str + " data.")
    end do
end
