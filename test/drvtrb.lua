

local json = require 'json'
local host = require 'host'
local util = require 'util'


local RunArgs = {
   N       = 16,
   id      = "test",
   shen    = false,
   zeta    = 1.0,   -- driving vorticity (1=full)
   cool    = "E4",
   B0      = 1e10,  -- B-field, in Gauss
   D0      = 1.00,  -- density, code units
   P0      = 0.05,  -- pressure, code units
   CFL     = 0.24,
   tmax    = 72.0,
   cpi     = 1.0,   -- checkpoint interval
   restart = "none",
   rmdiv   = false, -- run a div-clean on the input field
}
local StreamlinesFile

-- *****************************************************************************
-- Function to collect all available measurements from Mara
-- .............................................................................
local function Measurements(Status)

   local meas = { }

   meas.U                    = measure_mean_cons():astable()
   meas.P                    = measure_mean_prim():astable()

   meas.energies             = measure_mean_energies()
   meas.mean_velocity        = measure_mean_velocity():astable()

   meas.mean_T, meas.max_T   = measure_mean_max_temperature()
   meas.mean_B, meas.max_B   = measure_mean_max_magnetic_field()
   meas.mean_Ms, meas.max_Ms = measure_mean_max_sonic_mach()
   meas.mean_Ma, meas.min_Ma = measure_mean_min_alfvenic_mach()
   meas.max_lorentz_factor   = measure_max_lorentz_factor()

   meas.Status = util.deepcopy(Status)

   return meas
end


local function HandleErrors(Status, attempt)

   if attempt == 0 then -- healthy time-step
      set_godunov("plm-muscl", 2.0, 0)
      set_riemann("hlld")
      Status.Timestep = 1.0 * Status.Timestep
      return 0
   elseif attempt == 1 then
      set_godunov("plm-muscl", 1.5, 0)
      diffuse(0.2)
      Status.Timestep = 0.5 * Status.Timestep
      return 0
   elseif attempt == 2 then
      set_godunov("plm-muscl", 1.0, 0)
      diffuse(0.2)
      Status.Timestep = 0.5 * Status.Timestep
      return 0
   elseif attempt == 3 then
      set_godunov("plm-muscl", 0.0, 0)
      set_riemann("hll")
      diffuse(0.2)
      Status.Timestep = 0.5 * Status.Timestep
      return 0
   elseif attempt == 4 then
      diffuse(0.2)
      Status.Timestep = 0.5 * Status.Timestep
      return 0
   elseif attempt == 5 then
      diffuse(0.2)
      Status.Timestep = 0.5 * Status.Timestep
      return 0
   else
      return 1
   end
end


-- *****************************************************************************
-- Main driver, operates between checkpoints and then returns
-- .............................................................................
local function RunSimulation(Status, Howlong)

   local t0 = Status.CurrentTime
   local attempt = 0

   while Status.CurrentTime - t0 < Howlong do

      if Status.Iteration % 10 == 0 then
         local S = streamline({0.0, 0.0, 0.0}, 3.0, 1e-3, "magnetic")
         h5_open_file(StreamlinesFile, "r+")
         h5_write_array(string.format("mag%04d", Status.Iteration), S)
         h5_close_file()
         collectgarbage()
      end

      -- Measurements are made at the beginning of the timestep
      -- .......................................................................
      if Status.Iteration % 10 == 0 then
	 MeasureLog[Status.Iteration] = Measurements(Status)
      end


      -- 'attempt' == 0 when the previous iteration completed without errors
      -- .......................................................................
      if HandleErrors(Status, attempt) ~= 0 then
	 return 1
      end
      attempt = attempt + 1


      local dt = Status.Timestep
      local kzps, errors = advance(dt)

      if errors == 0 then
         driving.Advance(dt)
         driving.Resample()

         print(string.format("%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)",
                             Status.Iteration, attempt-1, Status.CurrentTime, dt,
			     kzps, 1e6/8/(1e3*kzps)))
	 io.flush()

	 attempt = 0
         Status.Timestep = get_timestep(RunArgs.CFL)
         Status.CurrentTime = Status.CurrentTime + Status.Timestep
         Status.Iteration = Status.Iteration + 1
      end
   end
   return 0
end



local function CheckpointWrite(Status)

   Status.Checkpoint = Status.Checkpoint + 1
   local datadir = string.format("data/%s", RunArgs.id)
   local chkpt = string.format("%s/chkpt.%04d.h5", datadir, Status.Checkpoint)

   if mpi_get_rank() == 0 then

      local version = mara_version()
      local program = " "
      local f = io.open("drvtrb.lua", "r")
      if f then
	 program = f:read("*all")
	 f:close()
      end

      h5_open_file(chkpt, "w")
      h5_write_numeric_table("status", Status)
      h5_write_string("measure", json.encode(MeasureLog))
      h5_write_string("driving", json.encode(driving.Serialize()))
      h5_write_string("runargs", json.encode(RunArgs))
      h5_write_string("program", program)
      h5_write_string("version", version)
      h5_close_file()

   end

   write_prim(chkpt, host.CheckpointOptions)
end


local function CheckpointRead(chkpt)

   h5_open_file(chkpt, "r")
   local status = h5_read_numeric_table("status")
   local measure = json.decode(h5_read_string("measure"))
   local field = json.decode(h5_read_string("driving"))
   h5_close_file()

   set_driving(field)
   read_prim(chkpt, host.CheckpointOptions)

   if RunArgs.rmdiv then
      local mean0, max0 = measure_mean_max_divB()
      local prim = get_prim()
      prim.Bx, prim.By, prim.Bz = fft_helmholtz(prim.Bx, prim.By, prim.Bz)
      init_prim(prim)
      local mean1, max1 = measure_mean_max_divB()
      print("before div clean, mean/max:", mean0, max0)
      print(" after div clean, mean/max:", mean1, max1)
   end

   return status, measure
end


local function InitSimulation()

   for k,v in pairs(cmdline.opts) do
      if type(RunArgs[k]) == 'number' then
	 RunArgs[k] = tonumber(v)
      else
	 RunArgs[k] = v
      end
   end

   print("runtime arguments:")
   print("------------------")
   for k,v in pairs(RunArgs) do
      if type(v) == 'number' and (math.log10(v) > 2 or math.log10(v) < -2) then
	 print(k,string.format("%3.2e", v))
      else
	 print(k,v)
      end
   end

   local function pinit(x,y,z)
      local B0 = RunArgs.B0 * units.Gauss()
      local D0 = RunArgs.D0
      local P0 = RunArgs.P0
      return { D0, P0, 0, 0, 0, B0, 0.0, 0.0 }
   end

   local function do_units()
      local LIGHT_SPEED = 2.99792458000e+10 -- cm/s

      local Density = 1e13         -- gm/cm^3
      local V       = LIGHT_SPEED  -- cm/s
      local Length  = 1e2          -- cm
      local Mass    = Density * Length^3.0
      local Time    = Length / V
      set_units(Length, Mass, Time)
      units.Print()
   end

   local N = RunArgs.N
   set_domain({-0.5,-0.5,-0.5}, {0.5,0.5,0.5}, {N,N,N}, 8, 2)

   do_units()
   set_fluid("rmhd")
   set_boundary("periodic")
   set_riemann("hll")
   set_advance("single")
   set_godunov("plm-muscl", 2.0, 0)
   set_driving(new_ou_field(3, 0.01, RunArgs.zeta, 3, 12345))

   if     RunArgs.cool == "T4" then
      set_cooling("T4", 40.0, 100.0)
   elseif RunArgs.cool == "E4" then
      set_cooling("E4",  0.1, 100.0)
   end

   if RunArgs.shen then
      local tab = load_shen(host.ShenFile, 0.08, {1e12, 1e16}, {0.1, 200.0})
      set_eos("shen", tab)
   else
      set_eos("gamma-law", 4.0/3.0)
   end

   local Status = { }

   if RunArgs.restart == "none" then

      init_prim(pinit) -- start a new model from scratch
      MeasureLog = { }

      Status.CurrentTime = 0.0
      Status.Iteration   = 0
      Status.Checkpoint  = 0
      Status.Timestep    = 0.0
   else
      -- read an existing model from the disk
      Status, MeasureLog = CheckpointRead(RunArgs.restart)
   end

   if mpi_get_rank() == 0 then
      local datadir = string.format("data/%s", RunArgs.id)
      os.execute(string.format("mkdir -p %s", datadir))
      os.execute(host.Filesystem(datadir))

      StreamlinesFile = string.format("%s/stream.h5", datadir)
      h5_open_file(StreamlinesFile, "w")
      h5_close_file()
   end

   print_mara()
   return Status
end


local MeasureLog = { }
local Status = InitSimulation()

while Status.CurrentTime < RunArgs.tmax do
   local error = RunSimulation(Status, RunArgs.cpi)
   if error ~= 0 then
      print("exiting due to failures\n")
      break
   end
   CheckpointWrite(Status)
end

