



h5_open_file(cmdline.args[1], "r")

visual.open_window()

for i=0,1850,10 do
   local S = h5_read_array(string.format("mag%04d", i))
   visual.draw_lines3d(S)
   os.execute('sleep 0.1')
end


h5_close_file()
