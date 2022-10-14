import numpy as np
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv as BaseSimpleEnv
from pettingzoo.mpe._mpe_utils import rendering

class SimpleEnv(BaseSimpleEnv):
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(200, 200)
            # fix the camera position with initial positions
            all_poses = [entity.state.p_pos for entity in self.world.entities]
            cam_range = np.max(np.abs(np.array(all_poses))) + 1
            self.viewer.set_max_size(cam_range)

        # create rendering geometry
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        # all_poses = [entity.state.p_pos for entity in self.world.entities]
        # cam_range = np.max(np.abs(np.array(all_poses))) + 1
        # self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')