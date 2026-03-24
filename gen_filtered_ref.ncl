begin
    f = addfile("data/test/wind/era5_uv850_2025120100_2.5x2.5.nc", "r")
    u = f->u
    v = f->v
    
    ; shlowp and shhighp are part of NCL's Spherepack wrappers
    ; They are functions.
    ; shlowp(data, truncation): removes l > truncation
    ; shhighp(data, truncation): removes l < truncation? 
    ; Let's use tri_trunc for bandpass if available or manual.
    
    ; Correct usage of shaec/shsec as functions:
    ; a and b must be defined or received from shaec.
    ; Wait, NCL documentation says shaec is a function. 
    ; The error "before or near shaec" often means it's not found or misused.
    
    ; Let's try tri_trunc(data, truncation)
    u_low = tri_trunc(u, 42) ; Keep l <= 42
    u_filt = u_low - tri_trunc(u, 4) ; Keep l <= 42 and remove l <= 4 -> Keep [5, 42]
    
    v_low = tri_trunc(v, 42)
    v_filt = v_low - tri_trunc(v, 4)
    
    copy_VarMeta(u, u_filt)
    copy_VarMeta(v, v_filt)
    
    out_file = "data/test/wind/era5_uv850_2025120100_2.5x2.5_t5-42_ncl.nc"
    system("rm -f " + out_file)
    out = addfile(out_file, "c")
    out->u = u_filt
    out->v = v_filt
end
