from datetime import datetime

def log_print(text, logfile):
    ts = datetime.now().strftime('[%m-%d %H:%M:%S]')
    text = f'{ts}\t{text}'
    print(text)
    logfile.writelines(text+'\n')  
    logfile.flush() 

def dict2text(input_dict):
	text=''
	if 'Project' in input_dict.keys():
		text += 'Project: '+ input_dict['Project'] + '\n'
	if 'Experiment' in input_dict.keys():
		text += 'Experiment: '+ input_dict['Experiment'] + '\n\n'
	for key, value in input_dict.items():
		if key in ['Project','Experiment']:
			continue
		text += f'{key}: {value}\n'
	return text