load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

; General script to process ERA5 data for PyStormTracker tests.
; Usage: ncl 'dir_in="/path/to/data/"' 'dir_out="./output/"' 'date="2025120100"' 'res="0.25"' process_era5_0.25.ncl

begin
    ; --- Configuration with Defaults ---
    if (.not. isvar("dir_in")) then
        dir_in = "/home/albert/PyStormTracker-Data/"
    end if
    if (.not. isvar("dir_out")) then
        dir_out = "/home/albert/PyStormTracker/data/test/era5/"
    end if
    if (.not. isvar("date")) then
        date = "2025120100"
    end if
    if (.not. isvar("res")) then
        res = "0.25" ; Resolution string for filenames
    end if
    if (.not. isvar("lmax_trunc")) then
        lmax_trunc = 42
    end if

    grid_str = res + "x" + res
    
    file_msl = dir_in + "era5_msl_2025-2026_djf_" + grid_str + ".nc"
    file_uv = dir_in + "era5_uv850_2025-2026_djf_" + grid_str + ".nc"
    file_vo = dir_in + "era5_vo850_2025-2026_djf_" + grid_str + ".nc"

    ; 1. Load First Frames
    print("Processing " + grid_str + " for date " + date + "...")
    
    f_msl_src = addfile(file_msl, "r")
    msl_raw = f_msl_src->msl(0,:,:)
    
    f_uv_src = addfile(file_uv, "r")
    u_raw = f_uv_src->u(0,0,:,:)
    v_raw = f_uv_src->v(0,0,:,:)
    
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

    ; 2. Calculate VODV
    vo_calc = uv2vrF_Wrap(u_raw, v_raw)
    dv_calc = uv2dvF_Wrap(u_raw, v_raw)
    
    ; 3. Spectral Filtering (T42)
    ; Determine nlon for synthesis
    dims = dsizes(msl_raw)
    nlon = dims(1)
    
    a = shaeC(msl_raw)
    
    ; Low-pass (T0-42)
    a_t0 = a
    a_t0(:, (lmax_trunc+1):, :) = 0.0
    msl_t0 = shseC(a_t0, nlon)
    
    ; Band-pass (T5-42)
    a_t5 = a_t0
    a_t5(:, 0:4, :) = 0.0
    msl_t5 = shseC(a_t5, nlon)

    ; 4. Save and Standardize Metadata
    
    ; Helper to clean and save variables
    procedure clean_save(var_in, name, long_name, units, std_name, suffix)
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
        
        out_name = dir_out + "era5_" + suffix + "_" + date + "_" + grid_str + ".nc"
        if (suffix .eq. "vodv850_ncl" .or. suffix .eq. "msl_t0-42_ncl" .or. suffix .eq. "msl_t5-42_ncl") then
             ; These are special cases handled differently or already grouped
        else
            system("rm -f " + out_name)
            f_o = addfile(out_name, "c")
            f_o->$name$ = var
            delete(f_o)
        end if
    end

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
    clean_save(vo_raw, "vo", "Vorticity (relative)", "s**-1", "atmosphere_relative_vorticity", "vo850")

    ; --- VODV (calculated) ---
    vo_c = vo_calc
    dv_c = dv_calc
    delete_VarAtts(vo_c, -1)
    vo_c@long_name = "relative vorticity"
    vo_c@units = "s**-1"
    vo_c@standard_name = "atmosphere_relative_vorticity"
    vo_c!0 = "latitude"
    vo_c!1 = "longitude"
    vo_c&latitude = lat
    vo_c&longitude = lon
    
    delete_VarAtts(dv_c, -1)
    dv_c@long_name = "divergence"
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

    ; --- MSL Raw ---
    clean_save(msl_raw, "msl", "Mean sea level pressure", "Pa", "air_pressure_at_mean_sea_level", "msl")

    ; --- MSL Filtered ---
    clean_save(msl_t0, "msl", "Mean sea level pressure", "Pa", "air_pressure_at_mean_sea_level", "msl_t0-42_ncl")
    f_t0_out = dir_out + "era5_msl_" + date + "_" + grid_str + "_t0-42_ncl.nc"
    system("rm -f " + f_t0_out)
    out_t0 = addfile(f_t0_out, "c")
    out_t0->msl = msl_t0
    delete(out_t0)

    msl_t5_f = tofloat(msl_t5)
    clean_save(msl_t5_f, "msl", "Mean sea level pressure", "Pa", "air_pressure_at_mean_sea_level", "msl_t5-42_ncl")
    f_t5_out = dir_out + "era5_msl_" + date + "_" + grid_str + "_t5-42_ncl.nc"
    system("rm -f " + f_t5_out)
    out_t5 = addfile(f_t5_out, "c")
    out_t5->msl = msl_t5_f
    delete(out_t5)

    print("Successfully processed " + grid_str + " data.")
end
