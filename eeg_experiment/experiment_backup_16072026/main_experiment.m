function main_experiment(subject)

parameter.subject_code = subject;
parameter.debugging = 0;
parameter.save_path = '/data/tu_bruera/prova/';

%% Screen Configurations
parameter.cross_size                                    = 100;
parameter.cross_color                                   = 220;
parameter.screen_refresh_rate                           = 60;
parameter.instruction_size                              = 13;
parameter.exp_size                              = 16;
parameter.instruction_color                             = 220;
parameter.screen_color                                  = 30;%10;%pilot with old beamer 30;
parameter.target_detection_zahlen_target_size           = 25;
parameter.target_detection_zahlen_target_color          = 220;

%% set up visual settings
rng('shuffle');

%% hardware preparation
%% visual display
screen_color    = parameter.screen_color;
Screen('Preference', 'SkipSyncTests', 1)
% visual parameter
instruction_size    = parameter.instruction_size;
exp_size    = parameter.exp_size;
instruction_color   = parameter.instruction_color;
cross_size          = parameter.cross_size;
cross_color         = parameter.cross_color;

% Open a window


AssertOpenGL;
screenNum       = 0;
%Screen('Preference','DefaultFontName','Helvetica');

if parameter.debugging ==1
    % debug window
    [my_window, rect] = Screen('OpenWindow', screenNum, [screen_color screen_color screen_color], [0,0,400,400]);
else
    % experiment window
    [my_window, rect] = Screen('OpenWindow', screenNum, [screen_color screen_color screen_color]);
    HideCursor;
end
parameter.rect      = rect;
parameter.my_window = my_window;

% Set Debuglevel to 3
Screen('Preference', 'visualDebuglevel', 3);

% skip the screen synchronization test
AssertOpenGL;
Screen('Preference', 'SkipSyncTests', 1);

% Set Priority level
priorityLevel = MaxPriority(my_window);
Priority(priorityLevel);
csvname = [parameter.save_path,filesep,'main_exp', filesep, subject,'_mainexp', '.csv'];

%if ismac == 1
%    while KbCheck end
%else
%    while KbCheck end
%end
KbCheck
% Set up the trigger channel 
global ioObj
ioObj = io32;
status = io32(ioObj);
global address
address = hex2dec('378');
global trigger_duration
trigger_duration = 0.01;

%set trigger to baseline
io32(ioObj, address, 0);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

first_welcome = ['Willkommen!\n\n\n'...
    'In dieser Studie untersuchen wir die Bedeutung von deutschen Wörtern \n'...
    'und deren Bezug zu Merkmalen der Wahrnehmung und Handlung.\n\n\n'...
    '[Drücken Sie 1, 2 oder 3, um weiterzumachen]']

print_instruction(my_window, rect,first_welcome, instruction_size, instruction_color);

second_welcome = ['Dieses Experiment besteht aus 13 kurzen Abschnitten.\n'...
                   'In jedem Abschnitt lesen Sie zunächst ein Wort und beantworten anschließend \n'...
                   'eine einfache Frage zu dessen Bedeutung mithilfe der Tasten 1, 2 oder 3 auf Ihrer Tastatur. \n\n'...
                   'Dies wird für einige Wörter wiederholt. Danach folgt eine kurze Pause, \n'...
                   'in der Sie sich entspannen können, bevor es mit dem nächsten Abschnitt weitergeht.\n\n\n'...
                   '[Drücken Sie 1, 2 oder 3, um weiterzumachen]']
    
print_instruction(my_window, rect,second_welcome, instruction_size, instruction_color);

third_welcome = ['In jedem Abschnitt wird Ihnen dieselbe Frage gestellt, die Sie dazu anregt,\n'... 
'über die Bedeutung der verwendeten Wörter aus einer bestimmten Perspektive nachzudenken.\n'...
'Zum Beispiel könnten wir Ihnen in einem Abschnitt folgende Frage zu dem Wort "Hund" stellen:\n\n'...
'"Wie stark denken Sie bei diesem Wort an etwas, das laut klingt?"\n[1=niedrig, 2=mittel, 3=hoch]\n\n'...
'und im nächsten Abschnitt dann die Frage:\n\n'...
'"Wie stark denken Sie bei diesem Wort an etwas, das sich langsam bewegt?"\n[1=niedrig, 2=mittel, 3=hoch]\n\n\n'...
'[Drücken Sie 1, 2 oder 3, um weiterzumachen]']

print_instruction(my_window, rect, third_welcome, instruction_size, instruction_color);

fourth_welcome = ['Es gibt keine richtigen oder falschen Antworten; \n'...
                  'Sie sollten je nach Ihren persönlichen Erfahrungen mit den Dingen, \n'...
                  'auf die sich die Wörter beziehen, antworten \n\n'...
                  '(zum Beispiel ist für mich ein Hund laut [und ich würde bei der ersten Frage 3 drücken], \n'...
                  'aber nicht langsam [deshalb würde ich bei der zweiten Frage 1 drücken]).\n\n\n'...
                  '[Drücken Sie 1, 2 oder 3, um weiterzumachen]']

print_instruction(my_window, rect, fourth_welcome, instruction_size, instruction_color);

fifth_welcome = ['Wir bitten Sie, die Frage schnell, aber präzise zu beantworten - und, kein Stress! :) \n'...
                 'Daher schlagen wir vor, dass Sie beim Lesen des Wortes bereits an den damit bezeichneten Gegenstand denken\n'...
                 ' und zwar im Hinblick auf die kommende Frage (zum Beispiel: Ist er laut?).\n\n'...
                 '...aber Vorsicht! Manchmal ist die Reihenfolge der Tasten umgekehrt (1=hoch, 2=mittel, 3=niedrig).\n\n\n'...
                '[Drücken Sie 1, 2 oder 3, um weiterzumachen]']

print_instruction(my_window, rect,fifth_welcome, instruction_size, instruction_color);

sixth_welcome = ['Bevor es richtig losgeht, gibt es ein paar Probedurchläufe, \n'...
                 'damit Sie das Experiment ausprobieren können!\n\n\n'...
                 '[Drücken Sie 1, 2 oder 3, um die Testphase zu starten.]'
                 ]
 print_instruction(my_window, rect,sixth_welcome, instruction_size, instruction_color);

%set trigger to baseline
io32(ioObj, address, 0);

trial_ws =  {'Gras', 'Konversation', 'Schnee'}
trial_qs = {'Wie sehr denken Sie bei diesem Wort an etwas, das extreme Temperatur hat, also sich besonders heiß oder kalt anfühlt?', 'Wie stark denken Sie bei diesem Wort an etwas, das Sie motiviert, etwas zu tun?'}
trial_ds ={'Temperatur', 'Motivation'}
trial_keys = {'[1=hoch      2=mittel      3=niedrig]', '[1=niedrig      2=mittel      3=hoch]'}

start = 0

for q  = [1:2]
    curr_q = trial_qs(q)
    curr_d = trial_ds(q)
    req_keys = trial_keys(q)
    ques = [char(curr_q)...
            '\n\n\n\n\n'...
            char(curr_d)...
            '?\n\n'...
            char(req_keys)... 
            '\n\n\n\n\n\n'...
            '[Drücken Sie 1, 2 oder 3, um den Abschnitt zu beginnen.]']
    print_instruction(my_window, rect, ques, instruction_size, instruction_color);
    for w = [1:3]
        curr_w = char(trial_ws(w))
        dim_ques = [char(curr_d)...
            '?\n\n'...
            char(req_keys)... 
            ] 
        print_word(my_window, rect, curr_w, exp_size, instruction_color, 0, ioObj, address, trigger_duration);
        print_question(my_window, rect, dim_ques, exp_size, instruction_color, 0, start, char(curr_d), char(req_keys), ioObj, address, trigger_duration);
    end
end

break_text = ['Pause! yay! \n\n\n'...
            'Ende von die Probedurchläufe\n\n\n:)'];
print_break(my_window, rect, break_text, instruction_size, instruction_color);

seventh_welcome = ['Bereit? \n\n\n\n'...
                 '[Drücken Sie 1, 2 oder 3, um das Experiment zu beginnen.]'
                 ]
print_instruction(my_window, rect,seventh_welcome, instruction_size, instruction_color);

sub_bids = ['sub-' num2str(str2num(subject), '%02d')];

for run_id = [1:13]
    io32(ioObj, address, 0);
    run_bids = ['run-' num2str(run_id, '%02d')];
    out_folder = {['experiment_logs/' sub_bids '/' run_bids]}
    out_path = strjoin([out_folder,'/',sub_bids,'_',run_bids,'_events.txt'], '');
    dim_ques = readtable(['randomized_exp_per_sub/',num2str(str2num(subject), '%02d'),'_',num2str(run_id, '%02d'),'_dq.mp3'], 'Delimiter','\t', 'FileType', 'text');
    dim = char(dim_ques.Var1);
    req_keys = char(dim_ques.Var3);
    trigger_dim = dim_ques.Var4;
    %%% question appears only at the beginning; then for questions we only
    %%% put the dimension
    ques = [char(dim_ques.Var2)...
            '\n\n\n\n\n'...
            dim...
            '?\n\n'...
            req_keys '\n\n\n\n\n\n'...
            '[Drücken Sie 1, 2 oder 3, um den Abschnitt zu beginnen.]']
    ques = strrep(ques, 'Ã¼', 'ü')
    ques = strrep(ques, 'Ã¤', 'ä')
    ques = strrep(ques, 'Ã¶', 'ö')
    print_instruction(my_window, rect, ques, exp_size, instruction_color);
    dim_ques = [dim...
            '?\n\n'...
            req_keys... 
            ]   
    dim_ques = strrep(dim_ques, 'Ã¼', 'ü')
    dim_ques = strrep(dim_ques, 'Ã¤', 'ä')
    dim_ques = strrep(dim_ques, 'Ã¶', 'ö')

    words = readtable(['randomized_exp_per_sub/',num2str(str2num(subject), '%02d'),'_',num2str(run_id, '%02d'),'_w.mp3'], 'Delimiter','\t', 'FileType', 'text');
    run_answers = [{'onset', 'duration', 'trial_type', 'trigger', 'pressed_key'}];
    for word_idx = [1:64];
        io32(ioObj, address, 0);
        word = char(words.Var1(word_idx));
        word = strrep(word, 'Ã¼', 'ü')
        word = strrep(word, 'Ã¤', 'ä')
        word = strrep(word, 'Ã¶', 'ö')
        %word = strrep(word, , 'ö')
        word_trigger = words.Var2(word_idx)
        start = GetSecs()
        tic
        print_word(my_window, rect, word, exp_size, instruction_color, word_trigger, ioObj, address, trigger_duration);
        stop = toc
        word_line = {num2str(start), num2str(stop), word, num2str(word_trigger), 'NA'};
        run_answers = [run_answers; word_line]
        start = GetSecs()
        answer = print_question(my_window, rect, dim_ques, exp_size, instruction_color, trigger_dim, start, dim, req_keys, ioObj, address, trigger_duration);
        run_answers = [run_answers; answer];

    end

    writetable(cell2table(run_answers), out_path, 'Delimiter','\t', 'FileType', 'text');
    
    %breaks between blocks
    break_text = ['Pause! yay! \n\n\n'...
            'Ende von Abschnitt' num2str(run_id) '\n\n\n:)'];
    print_break(my_window, rect,break_text, exp_size, instruction_color);

    continue_text = ['Drücken Sie 1, 2 oder 3, um weiterzumachen'];
    print_instruction(my_window, rect,continue_text, exp_size, instruction_color);
  
end

end_text = ['The end!\n\n yay yay yay! \n\n\n'...
            'Vielen Dank für Ihre Teilnahme an diesem Experiment!\n\n\n'...
            'Bitte Drücken Sie 1, 2 oder 3'];
print_break(my_window, rect, end_text, exp_size, instruction_color);

% release keyboard restrictions
%RestrictKeysForKbCheck([]);

ShowCursor;
Screen('CloseAll');
