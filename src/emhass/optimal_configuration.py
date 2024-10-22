
class Deferrable:

	def __init__(self, nominal_power, total_hours, start_timestep=0, end_timestep=0, constant = False, start_penalty = 0.0, as_semi_cont = True, load_config = False):
		self.power = nominal_power
		self.total_hours = total_hours
		self.start_timestep = start_timestep
		self.end_timestep = end_timestep
		self.constant = constant
		self.start_penalty = start_penalty
		self.as_semi_cont = as_semi_cont
		self.load_config = load_config

class OptimalConfiguration:

	def __init__(self, optim_conf:dict):
		self._as_dict = optim_conf
		self.deferrables = []
		for i in range(optim_conf['num_def_loads']):
			nominal_power = optim_conf['P_deferrable_nom'][i]
			total_hours = optim_conf['def_total_hours'][i]
			start_timestep = optim_conf['def_start_timestep'][i]
			end_timestep = optim_conf['def_end_timestep'][i]
			constant = optim_conf['set_def_constant'][i]
			start_penalty = optim_conf['def_start_penalty'][i]
			as_semi_cont = optim_conf['treat_def_as_semi_cont'][i]
			self.deferrables.append(Deferrable(nominal_power, total_hours, start_timestep, end_timestep, constant, start_penalty, as_semi_cont))
		self.lp_solver = optim_conf.get('lp_solver', 'default')
		self.lp_solver_path = optim_conf.get('lp_solver_path', 'empty')
		self.set_use_battery = optim_conf.get('set_use_battery', False)
		self.set_nocharge_from_grid = optim_conf.get('set_nocharge_from_grid', False)
		self.set_nocharge_from_grid = optim_conf.get('set_nocharge_from_grid', False)
		self.set_battery_dynamic = optim_conf.get('set_battery_dynamic', False)
		self.battery_dynamic_max = optim_conf.get('battery_dynamic_max', 100)
		self.battery_dynamic_min = optim_conf.get('battery_dynamic_min', 0)
		self.set_total_pv_sell = optim_conf.get('set_total_pv_sell', False)
		self.weight_battery_discharge = optim_conf.get('weight_battery_discharge', 0)
		self.weight_battery_charge = optim_conf.get('weight_battery_charge', 100)

	def add_deferrable(self, new):
		self.deferrables.append(new)

	def _as_dict(self):
		optim_conf = dict()
		optim_conf = self._as_dict['opt'].optim_conf
		optim_conf['num_def_loads'] = len(self.deferrables)
		optim_conf['P_deferrable_nom'] = [d.power for d in self.deferrables]
		optim_conf['def_total_hours'] = [d.total_hours for d in self.deferrables]
		optim_conf['def_start_timestep'] = [d.start_timestep for d in self.deferrables]
		optim_conf['def_end_timestep'] = [d.end_timestep for d in self.deferrables]
		optim_conf['set_def_constant'] = [d.constant for d in self.deferrables]
		optim_conf['def_start_penalty'] = [d.start_penalty for d in self.deferrables]
		optim_conf['def_start_penalty'] = [d.start_penalty for d in self.deferrables]
		optim_conf['def_load_config'] = [d.load_config for d in self.deferrables]
		optim_conf['treat_def_as_semi_cont'] = [d.as_semi_cont for d in self.deferrables]
		input_data_dict['opt'].optim_conf = optim_conf
	

