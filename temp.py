class Traj_():
    ...
    
        def check_intersect_prim(self, geom, RV_old, RV_new, invisible_wall_x): 
        #check if out of bounds for passing to aim
        if RV_new[0, 0] > invisible_wall_x and RV_new[0, 1] < 1.2:
            return True  # self.print_log('primary hit invisible wall, r = %s' % str(np.round(r, 3)))

        if geom.check_chamb_intersect('prim', RV_old[0, 0:3], RV_new[0, 0:3]):
            self.IntersectGeometry['chamb'] = True
            return True  # self.print_log('Primary intersected chamber entrance')

        plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3], RV_new[0, 0:3])
        if plts_flag:
            self.IntersectGeometry[plts_name] = True
            return True   # #self.print_log('Primary intersected ' + plts_name + ' plates')

        if geom.check_fw_intersect(RV_old[0, 0:3], RV_new[0, 0:3]): 
            return True  # stop primary trajectory calculation  # self.print_log('Primary intersected first wall')

        return False

    #%%----------------------------------------------------------------------------
    def pass_prim(self, E_interp, B_interp, geom, tmax=1e-5, 
                  invisible_wall_x=5.5):
        '''
        passing primary trajectory from initial point self.RV0
        E_interp : dictionary with E field interpolants
        B_interp : list with B fied interpolants
        geom : Geometry object
        '''
        self.reset_flags("only_intersection_flags")
        t = 0.
        dt = self.dt1
        RV_old = self.RV0  # initial position
        RV = self.RV0  # array to collect all r, V
        k = self.q / self.m
        tag_column = [10]

        while t <= tmax:
            r = RV_old[0, :3]
            E_local = return_E(r, E_interp, self.U, geom)
            B_local = return_B(r, B_interp); 
            if np.isnan(B_local).any(): break # self.print_log('Btor is nan, r = %s' % str(r))
            # *********************************************************************
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)
            # *********************************************************************
            RV, tag_column =  np.vstack((RV, RV_new)),  np.hstack((tag_column, 10))
            if self.check_intersect_prim(geom, RV_old, RV_new, invisible_wall_x): break  
            RV_old, t = RV_new, t + dt
            
        self.RV_prim, self.tag_prim = RV, tag_column
        
    #-----------------------------------------------------------------------------
    
    def check_aim(self, RV_new, r_aim, eps_xy=1e-3, eps_z=1e-3): 
        # check XY plane:    
        if (np.linalg.norm(RV_new[0, :2] - r_aim[:2]) <= eps_xy): 
            self.IsAimXY = True   # print('aim XY!')
        # check XZ plane:
        if (np.linalg.norm(RV_new[0, [0, 2]] - r_aim[[0, 2]]) <= eps_z): 
            self.IsAimZ = True    # print('aim Z!')    

def ___No_break___(): 
    pass

    def check_intersect_sec(self, geom, RV_old, RV_new, invisible_wall_x): 
        # #check if out of bounds for passing to aim
        if RV_new[0, 0] > invisible_wall_x:
            return True, None  # self.print_log('secondary hit invisible wall, r = %s' % str(np.round(r, 3)))
            
        if geom.check_chamb_intersect('sec', RV_old[0, 0:3], RV_new[0, 0:3]):
            self.IntersectGeometrySec['chamb'] = True  # print('Secondary intersected chamber exit')
            ___No_break___ # ??? Why

        plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3], RV_new[0, 0:3])
        if plts_flag:
            self.IntersectGeometrySec[plts_name] = True # self.print_log('Secondary intersected ' + plts_name + ' plates')
            ___No_break___ # ??? Why

        # find last point of the secondary trajectory
        if (RV_new[0, 0] > 2.5) and (RV_new[0, 1] < 1.5): # if Last point is outside
            # intersection with the stop plane:
            r_intersect = line_plane_intersect(stop_plane_n, r_aim, RV_new[0, :3]-RV_old[0, :3], RV_new[0, :3])
            
            # check if r_intersect is between RV_old and RV_new:
            if is_between(RV_old[0, :3], RV_new[0, :3], r_intersect):
                #RV_new[0, :3] = r_intersect
                #RV = np.vstack((RV, RV_new))
                # tag_column = np.hstack((tag_column, ??)) ????????? Не добавлен таг. Забыт или так надо?  
                self.IsAimXY, self.IsAimZ = self.check_aim(RV_new, r_aim, eps_xy, eps_z)  # check XY plane, check XZ plane
                return True, r_intersect
          return False, None  


    #%%-----------------------------------------------------------------------------
    def pass_sec(self, RV0, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1., 0., 0.]), tmax=5e-5,
                 eps_xy=1e-3, eps_z=1e-3, invisible_wall_x=5.5):
        '''
        passing secondary trajectory from initial point RV0 to point r_aim
        with accuracy eps
        RV0 : initial position and velocity
        '''
        # print('Passing secondary trajectory')
        self.reset_flags("all")

        t = 0.
        dt = self.dt2
        RV_old = RV0  # initial position
        RV = RV0  # array to collect all [r,V]
        tag_column = [20]
        k = 2*self.q / self.m

        while t <= tmax:  # to witness curls initiate tmax as 1 sec
            r = RV_old[0, :3]
            E_local = return_E(r, E_interp, self.U, geom)
            B_local = return_B(r, B_interp); 
            if np.isnan(B_local).any(): break  # self.print_log('Btor is nan, r = %s' % str(r))
            # ******************************************************
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)
            # ****************************************************** 
        
            stop, r_intersect = self.check_intersect_sec(geom, RV_old, RV_new, invisible_wall_x)
            if r_intersect in not None: 
                RV_new[0, :3] = r_intersect
                RV = np.vstack((RV, RV_new))
                # tag_column = np.hstack((tag_column, ??)) ????????? Не добавлен таг. Забыт или так надо?                  
            if stop: 
                break
        
            # continue trajectory calculation:
            RV_old, t = RV_new, t + dt
            RV, tag_column = np.vstack((RV, RV_new)),   np.hstack((tag_column, 20))

        self.RV_sec, self.tag_sec = RV, tag_column

    #%%-----------------------------------------------------------------------------
            
    def pass_fan(self, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1., 0., 0.]), eps_xy=1e-3, eps_z=1e-3,
                 no_intersect=False, no_out_of_bounds=False, 
                 invisible_wall_x=5.5):
        '''
        passing fan from initial point self.RV0
        '''
        # ********************************************************* #               
        self.pass_prim(E_interp, B_interp, geom, invisible_wall_x=invisible_wall_x)
        # ********************************************************* #               
                       
        # create a list fro secondary trajectories:
        list_sec = []
        # check intersection of primary trajectory:
        if True in self.IntersectGeometry.values():
            self.Fan = [] # list_sec   # print('Fan list is empty')
            return

        # check eliptical radius of particle:  # 1.5 m - major radius of a torus, elon - size along Y
        mask = np.sqrt((self.RV_prim[:, 0] - geom.R)**2 + (self.RV_prim[:, 1] / geom.elon)**2) <= geom.r_plasma   
        self.tag_prim[mask] = 11

        # list of initial points of secondary trajectories:
        RV0_sec = self.RV_prim[(self.tag_prim == 11)]

        for RV02 in RV0_sec:
            RV02 = np.array([RV02])
            # ********************************************************* #
            self.pass_sec(RV02, r_aim, E_interp, B_interp, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, invisible_wall_x=invisible_wall_x)
            # ********************************************************* #
            if not (     (no_intersect and True in self.IntersectGeometrySec.values()) or (no_out_of_bounds and self.B_out_of_bounds)    ):
                list_sec.append(self.RV_sec)


        self.Fan = list_sec
        
    #%%-----------------------------------------------------------------------------


    def check_fan_twisted(self, r_aim):
        signs = np.array([np.sign(np.cross(RV[-1, :3], r_aim)[-1]) for RV in self.Fan])
        are_higher = np.argwhere(signs == -1)
        are_lower = np.argwhere(signs == 1)
        twisted_fan = False  # flag to detect twist of the fan

        if are_higher.shape[0] == 0:
            n = int(are_lower[are_lower.shape[0]//2]) # print('all secondaries are lower than aim!')
        elif are_lower.shape[0] == 0:
            n = int(are_higher[are_higher.shape[0]//2]) # print('all secondaries are higher than aim!')
        else:
            if are_higher[-1] > are_lower[0]:
                twisted_fan = True  # print('Fan is twisted!')
                n = int(are_lower[-1])
            else:
                n = int(are_higher[-1])  # find the last one which is higher
                self.fan_ok = True
        return n, twisted_fan

    def reset_flags(self, options): 
        if (options == "all") or (options == "intersection_flags_and_aim"): 
            self.IsAimXY = False
            self.IsAimZ = False
            
        if (options == "all"): 
            self.B_out_of_bounds = False
        
        if (options == "only_intersection_flags") or (options == "all") or (options == "intersection_flags_and_aim"): 
            # reset intersection flags for secondaries
            for key in self.IntersectGeometrySec.keys():
                self.IntersectGeometrySec[key] = False
    
    def skip_pass_to_target(self):
        if True in self.IntersectGeometry.values(): 
            return True # print('There is intersection at primary trajectory'); 
        elif len(self.Fan) == 0: 
            return True # print('NO secondary trajectories'); 
        else:
            return False        
    
    def find_the_index_of_the_point_in_primary_traj_closest_to(self, RV_new): 
        # insert RV_new into primary traj
        # find the index of the point in primary traj closest to RV_new
        ind = np.nanargmin(np.linalg.norm(self.RV_prim[:, :3] - RV_new[0, :3], axis=1))
        if is_between(self.RV_prim[ind, :3], self.RV_prim[ind+1, :3], RV_new[0, :3], eps=1e-4):
            i2insert = ind+1
        else:
            i2insert = ind    
        return i2insert 
    
    #%%-----------------------------------------------------------------------------
    def pass_to_target(self, r_aim, E_interp, B_interp, geom,
                       stop_plane_n=np.array([1., 0., 0.]),
                       eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10,
                       no_intersect=False, no_out_of_bounds=False, 
                       invisible_wall_x=5.5):
        '''
        find secondary trajectory which goes directly to target
        '''
        if self.skip_pass_to_target(): return  
        self.reset_flags("intersection_flags_and_aim")

        # find which secondaries are higher/lower than r_aim   # sign = -1 means higher, 1 means lower        
        n, twisted = self.check_fan_twisted(r_aim)  
        RV_old = np.array([self.Fan[n][0]])

        # find secondary, which goes directly into r_aim
        self.dt1 = self.dt1/2.
        while True:
            # make a small step along primary trajectory
            r = RV_old[0, :3]
            B_local = return_B(r, B_interp);   
            if np.isnan(B_local).any(): break
            E_local = np.array([0., 0., 0.])
            # ********************************************************* #
            RV_new = runge_kutt(self.q / self.m, RV_old, self.dt1, E_local, B_local)                                   
            # pass new secondary trajectory
            # ********************************************************* #
            self.pass_sec(RV_new, r_aim, E_interp, B_interp, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, invisible_wall_x=invisible_wall_x)
            # ********************************************************* #               

            # check XY flag
            if self.IsAimXY:
                # insert RV_new into primary traj
                i2insert = self.find_the_index_of_the_point_in_primary_traj_closest_to(RV_new)
                self.RV_prim, self.tag_prim = np.insert(self.RV_prim, i2insert, RV_new, axis=0),   np.insert(self.tag_prim, i2insert, 11, axis=0)
                break

            # check if the new secondary traj is lower than r_aim
            if (not twisted and np.sign(np.cross(self.RV_sec[-1, :3], r_aim)[-1]) > 0): # if lower, halve the timestep and try once more
                self.dt1 = self.dt1/2.   # print('dt1={}'.format(self.dt1))
                if self.dt1 < dt_min:                    
                    break # print('dt too small')
            else:
                # if higher, continue steps along the primary
                RV_old = RV_new


#%%-----------------------------------------------------------------------------
def optimize_B2(tr, geom, UB2, dUB2, E, B, dt, stop_plane_n, target='aim',
                optimize=True, eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10):
    '''
    get voltages on B2 plates and choose secondary trajectory
    which goes into target
    '''
    # set up target
    print('Target: ' + target)
    r_aim = geom.r_dict[target]
    attempts_opt = 0
    attempts_fan = 0
    while True:
        tr.U['B2'], tr.dt1, tr.dt2 = UB2, dt, dt

        # pass fan of secondaries
        # ********************************************************* #               
        tr.pass_fan(r_aim, E, B, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, no_intersect=True, 
            no_out_of_bounds=True, invisible_wall_x=geom.r_dict[target][0]+0.2)
        # pass trajectory to the target
        # ********************************************************* #               
        tr.pass_to_target(r_aim, E, B, geom, stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z, dt_min=dt_min, no_intersect=True, 
            no_out_of_bounds=True, invisible_wall_x=geom.r_dict[target][0]+0.2)
        # ********************************************************* #               

        print('IsAimXY = ', tr.IsAimXY); print('IsAimZ = ', tr.IsAimZ)
        if True in tr.IntersectGeometry.values():
            break
        if not tr.fan_ok:
            attempts_fan += 1
        if attempts_fan > 3 or len(tr.Fan) == 0:
            print('Fan of secondaries is not ok')
            break

        if optimize:
            # change UB2 value proportional to dz
            if not tr.IsAimZ:
                dz = r_aim[2]-tr.RV_sec[-1, 2]
                print('UB2 OLD = {:.2f}, z_aim - z = {:.4f} m'.format(UB2, dz))
                
                UB2_old = UB2 
                UB2 = UB2 - dUB2*dz
                if np.isnan(UB2): 
                    tr.print_log("dUB2 = %f" % dUB2); tr.print_log("dz = %f" % dz); tr.print_log("UB2_old = %f" % UB2_old)
                    
                print('UB2 NEW = {:.2f}'.format(UB2))
                attempts_opt += 1
            else:
                break
            # check if there is a loop while finding secondary to aim
            if attempts_opt > 20:
                print('too many attempts B2!')
                break
        else:
            print('B2 was not optimized')
            break
    return tr
    
'''    
pass_fan         calls    pass_prim    
pass_to_target   calls    pass_sec
optimize_B2      calls    pass_fan   pass_to_target
'''
