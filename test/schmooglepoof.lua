-- Set up Polytrope as initial dnesity with function sin(r)/r
--Try to run it!
-- remove samps?

local json = require 'json'
local host = require 'host'
local util = require 'util'

local RunArgs = {
   N       = 32,
   id      = "test",
   source  = "gravity",  
   D0      = 10.00,   -- central density constant, code units
   K       = 1.0,   -- K = P0/D0
   ALF     = .07,   -- ALF = K/(2*pi*G)
   CFL     = 0.24,
   sepstar = 3,
   tmax    = .01,
   cpi     = .003,    -- checkpoint interval, in code time units
   restart = "none",
   shen    = false,
   eosfile = "none", -- i.e. nseos.h5
   fluid   = "euler", -- euler, srhd, rmhd
   samps   = 0,       -- samps > 0 indicates logging `samps` points per timestep
   advance   = "rk3",
   riemann   = "hllc",
   godunov   = "weno-split"
}

math.randomseed(mpi_get_rank()) -- for random point sampling
local NumberOfConserved = { euler=5, srhd=5, rmhd=8 }
local NumberOfGhostZones = 3


-- *****************************************************************************
-- A module to manage collection and output of point samplings
-- .............................................................................
local PointsSampler = { }
PointsSampler.points = { }
PointsSampler.fname = "samples.h5" -- is over-written at startup
PointsSampler.cache_size = 500000
PointsSampler.samples_per_call = 1000

function PointsSampler:append_new_samples(Status)
   local start = os.clock()
   for n=1,self.samples_per_call do
      local x = math.random() - 0.5
      local y = math.random() - 0.5
      local z = math.random() - 0.5
      local P = prim_at_point{x,y,z}:astable()
      table.insert(self.points, Status.CurrentTime)
      table.insert(self.points, x)
      table.insert(self.points, y)
      table.insert(self.points, z)
      for _,v in pairs(P) do
	 table.insert(self.points, v)
      end
   end
   print(string.format("point sampling took %3.2f seconds",
		       os.clock() - start))
end
function PointsSampler:create_initial_file()
   if mpi_get_rank() == 0 then
      print("creating initial points file", self.fname)
      h5_open_file(self.fname, "w")
      h5_close_file()
   end
end
function PointsSampler:purge(Status)
   local start = os.clock()
   local outarr = lunum.array(self.points)
   local dsetnm = string.format("samples-%05d-t%05d", mpi_get_rank(),
				Status.Iteration)
   local Nc = NumberOfConserved[RunArgs.fluid] + 4 -- since x^\mu is prepended
   outarr:resize{outarr:size()/Nc,Nc}
   self.points = { }
   collectgarbage()

   for i=0,mpi_get_size()-1 do
      if mpi_get_rank() == i then
	 h5_open_file(self.fname, "r+")
	 h5_write_array(dsetnm, outarr)
	 h5_close_file()
      end
      mpi_barrier()
   end
   print(string.format("purge samples took %3.2f seconds",
		       os.clock() - start))
end


-- *****************************************************************************
-- Function to collect all available measurements from Mara
-- .............................................................................
local function Measurements(Status)
   local meas = { }
   local dtmeas = Status.CurrentTime - Status.LastMeasurementTime

   --meas.U                    = measure_mean_cons():astable()
   --meas.P                    = measure_mean_prim():astable()

   meas.energies             = measure_mean_energies()
  --meas.mean_velocity        = measure_mean_velocity():astable()

   --meas.mean_T, meas.max_T   = measure_mean_max_temperature()
   --meas.mean_Ms, meas.max_Ms = measure_mean_max_sonic_mach()
   --meas.mean_Ma, meas.min_Ma = measure_mean_min_alfvenic_mach()

   --meas.Status = util.deepcopy(Status)
   return meas
end


local function HandleErrorsEuler(Status, attempt)

   set_advance("rk3")
   if attempt == 0 then -- healthy time-step
      set_godunov("weno-split")
      Status.Timestep = 1.0 * Status.Timestep
      return 0
--   elseif attempt == 1 then
--      Status.Timestep = 0.1 * Status.Timestep
--      return 0
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
   local ErrorHandlers = { euler=HandleErrorsEuler,
			   srhd=HandleErrorsSrhd,
			   rmhd=HandleErrorsRmhd }

   while Status.CurrentTime - t0 < Howlong do
      collectgarbage()

      local stopfname = string.format("data/%s/MARA_STOP", RunArgs.id)
      local stopfile = io.open(stopfname, "r")
      if stopfile then
	 print("Mara exiting, detected", stopfname)
	 stopfile.close()
	 mpi_barrier()
	 if mpi_get_rank() == 0 then
	    os.remove(stopfname)
	 end
	 return "STOP"
      end


      -- Measurements are made at the beginning of the timestep
      -- .......................................................................
      if Status.Iteration % 1 == 0 then
	 MeasureLog[Status.Iteration] = Measurements(Status)
      end

      if RunArgs.samps > 0 then
	 PointsSampler:append_new_samples(Status)
	 if #PointsSampler.points > PointsSampler.cache_size then
	    PointsSampler:purge(Status)
	 end
      end

      -- 'attempt' == 0 when the previous iteration completed without errors
      -- .......................................................................
      if ErrorHandlers[RunArgs.fluid](Status, attempt) ~= 0 then
	 return 1
      end
      attempt = attempt + 1


      local dt = Status.Timestep
      local kzps, errors = advance(dt)

      if errors == 0 then

	 local Nq = NumberOfConserved[RunArgs.fluid]
         print(string.format("%05d(%d): t=%5.4f dt=%5.4e %3.1fkz/s %3.2fus/(z*Nq)",
                             Status.Iteration, attempt-1, Status.CurrentTime, dt,
			     kzps, 1e6/Nq/(1e3*kzps)))
	 io.flush()

	 attempt = 0
         Status.Timestep = get_timestep(RunArgs.CFL)
         Status.CurrentTime = Status.CurrentTime + Status.Timestep
         Status.Iteration = Status.Iteration + 1
      end
   end
   return 0
end



local function CheckpointWrite(Status, OptionalName)

   Status.Checkpoint = Status.Checkpoint + 1
   local datadir = string.format("data/%s", RunArgs.id)
   local chkpt
   if OptionalName then
      chkpt = string.format("%s/chkpt.%s.h5", datadir, OptionalName)
   else
      chkpt = string.format("%s/chkpt.%04d.h5", datadir, Status.Checkpoint)
   end

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
   h5_close_file()

   read_prim(chkpt, host.CheckpointOptions)

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
      --Set up a Polytrope
      --local r = math.sqrt(x^2+y^2+z^2)
      local D0 = RunArgs.D0
      local K = RunArgs.K
      local ALF = RunArgs.ALF 
      local G = K * ALF ^ 2 / 2*math.pi
      local P0 = K*D0^2
      local DENS = D0
      local dr = (ALF*RunArgs.sepstar)
      local r = math.sqrt(x^2+y^2+z^2)
      local r1 = math.sqrt((x-dr)^2+y^2+z^2)
      local r2 = math.sqrt((x+dr)^2+y^2+z^2)

      if (r > 0.0) and ((r/ALF) < math.pi-1e-3) then
        DENS = D0 * math.sin(r/ALF)/(r/ALF) --]]
    --[[  if (r1 > 0) and (r1/ALF < math.pi-1e-3) then
        DENS = D0 * math.sin(r1/ALF)/(r1/ALF)
      elseif (r2 > 0) and (r2/ALF < math.pi-1e-3) then
         DENS = D0 * math.sin(r2/ALF)/(r2/ALF) --]]
      else
          DENS = 1e-1 -- Remove unphysical solution
      end

      local PRES = (DENS^2)*K
      
      return { DENS, PRES, 0, 0, 0 }
   end

   local N = RunArgs.N
   set_domain({-0.5,-0.5,-0.5}, {0.5,0.5,0.5}, {N,N,N},
	      NumberOfConserved[RunArgs.fluid], NumberOfGhostZones)

   local tabeos = require 'tabeos'
   tabeos.MakeNeutronStarUnits()
   set_fluid(RunArgs.fluid)
   set_boundary("periodic")
   config_solver({IS="sz10", sz10A=100.0})

   if     RunArgs.source == "gravity" then
      set_sources("gravity")
   end

   if RunArgs.eosfile ~= "none" then
      tabeos.LoadMicroPh(RunArgs.eosfile)
   else
      set_eos("gamma-law", 2)
   end

   local Status = { }

   if RunArgs.restart == "none" then
      init_prim(pinit) -- start a new model from scratch
      MeasureLog = { }

      Status.CurrentTime = 0.0
      Status.Iteration   = 0
      Status.Checkpoint  = 0
      Status.Timestep    = 0.0
      Status.LastMeasurementTime = 0.0
   else
      -- read an existing model from the disk
      Status, MeasureLog = CheckpointRead(RunArgs.restart)
      -- older checkpoints will not have LastMeasurementTime:
      if not Status.LastMeasurementTime then
	 Status.LastMeasurementTime = Status.CurrentTime
      end
   end

   boundary.ApplyBoundaries()

   local datadir = string.format("data/%s", RunArgs.id)
   if mpi_get_rank() == 0 then
      os.execute(string.format("mkdir -p %s", datadir))
      os.execute(host.Filesystem(datadir))
   end
   
   PointsSampler.samples_per_call = RunArgs.samps
   if RunArgs.samps > 0 then
      PointsSampler.fname = string.format("%s/samples.h5", datadir)
      PointsSampler:create_initial_file()
   end

   print_mara()
   return Status
end


local MeasureLog = { }
local Status = InitSimulation()

while Status.CurrentTime < RunArgs.tmax do
   local error = RunSimulation(Status, RunArgs.cpi)
   if error == "STOP" then
      print("exiting upon request\n")
      CheckpointWrite(Status, "stop")
      break
   elseif error ~= 0 then
      print("exiting due to failures\n")
      CheckpointWrite(Status, "fail")
      break
   end
   CheckpointWrite(Status)
end

