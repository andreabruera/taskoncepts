function print_word(my_window, rec, instruction, stimuli_size, stimuli_color, trigger, ioObj, address, trigger_duration)
stimuli_color = [stimuli_color stimuli_color stimuli_color];
%Screen(my_window,'TextFont','Helvetica');
Screen('TextSize',my_window, stimuli_size);

DrawFormattedText(my_window, '+', 'center', 'center', stimuli_color);
Screen(my_window, 'Flip');
%KbWait();

WaitSecs(.75);

DrawFormattedText(my_window, instruction, 'center', 'center', stimuli_color);

%DrawFormattedText(my_window, '... Weiter gehts per Tastendruck ...', 'center', rec(4)-rec(4)/7.5, stimuli_color);                            
Screen(my_window, 'Flip');
%KbWait();
io32(ioObj, address, trigger);
WaitSecs(trigger_duration);
io32(ioObj, address, 0);

WaitSecs(0.5);
DrawFormattedText(my_window, '+', 'center', 'center', stimuli_color);
Screen(my_window, 'Flip');
%KbWait();

WaitSecs(0.75);

end
