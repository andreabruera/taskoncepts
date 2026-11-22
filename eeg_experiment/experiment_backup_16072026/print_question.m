function answer = print_question(my_window, rec, instruction, stimuli_size, stimuli_color, trigger, start, dim, req_keys, ioObj, address, trigger_duration)
stimuli_color = [stimuli_color stimuli_color stimuli_color];
%Screen(my_window,'TextFont','Helvetica');
Screen('TextSize',my_window, stimuli_size);
DrawFormattedText(my_window, instruction, 'center', 'center', stimuli_color);

%DrawFormattedText(my_window, '... Weiter gehts per Tastendruck ...', 'center', rec(4)-rec(4)/7.5, stimuli_color);                            
Screen(my_window, 'Flip');
%KbWait();

%KbQueueCreate
%KbQueueStart
ok = 0;
tic
io32(ioObj, address, trigger);
WaitSecs(trigger_duration);
io32(ioObj, address, 0);
while ok == 0
    [keyIsDown, secs, keyCode, deltaSecs]= KbCheck();
    %pressed
    if keyIsDown
        kpressed = find(keyCode);
        %kpressed
        %keyCode
        %if ismember(kpressed(1),[11,12,13])
        if ismember(kpressed(1),[49, 50,51])
            %if ismember(keyCode,[11,12,13])
            presstime = num2str(toc);
            %presstime
            ok = 1;
            answer = {num2str(start), presstime, [dim ',' req_keys], num2str(trigger), num2str(kpressed)};
            %answer
        end
    end
end


end
