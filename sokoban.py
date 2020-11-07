#!/usr/bin/env python3

# Based on original implementation from:
# https://github.com/morenod/sokoban.git

from absl import app
from absl import flags

import IPython

import pygame
import pickle
import random
import signal
import skimage
import string
import sys
import time
import os

import numpy as np
from PIL import Image


class SessionEvent:

  def __init__(self, fovea, move, offset):
    self.fovea = fovea
    self.move = move
    self.offset = offset


class Session:

  def __init__(self):
    self.events = []


class Praxis:

  def __init__(self):
    self.sessions = []
    self.filename = 'praxis.pickle'
    if os.path.isfile(self.filename):
      self.sessions = pickle.load(open(self.filename, 'rb'))

  def __del__(self):
    if self.sessions:
      pickle.dump(self.sessions, open(self.filename, 'wb'))


class Game:

  FLOOR = ' '
  WALL = '#'
  WORKER = '@'
  DOCK = '.'
  BOX = '$'
  DOCKED_BOX = '*'
  DOCKED_WORKER = '+'

  def __init__(self, filename, level):
    self.matrix = []
    file = open(filename, 'r')
    level_found = False
    for line in file:
      row = []
      if not level_found:
        if "Level " + level == line.strip():
          level_found = True
      else:
        if line.strip() != "":
          row = []
          for c in line:
            if c != '\n' and self.is_valid_value(c):
              row.append(c)
            elif c == '\n':
              continue
            else:
              assert False, f"Invalid value detected in level {level}: {c}"
          self.matrix.append(row)
        else:
          break
    assert level_found, f'Failed to find level "{level}" in {filename}'

  def is_valid_value(self, char):
    return char in [
        Game.FLOOR, Game.WALL, Game.WORKER, Game.DOCK, Game.DOCKED_BOX,
        Game.BOX, Game.DOCKED_WORKER
    ]

  def load_size(self, gain=1):
    x = 0
    y = len(self.matrix)
    for row in self.matrix:
      if len(row) > x:
        x = len(row)
    return (x * gain, y * gain)

  def get_matrix(self):
    return self.matrix

  def print_matrix(self):
    for row in self.matrix:
      for char in row:
        sys.stdout.write(char)
        sys.stdout.flush()
      sys.stdout.write('\n')

  def get_content(self, x, y):
    return self.matrix[y][x]

  def set_content(self, x, y, content):
    if self.is_valid_value(content):
      self.matrix[y][x] = content
    else:
      print("ERROR: Value '" + content + "' to be added is not valid")

  def worker(self):
    x = 0
    y = 0
    for row in self.matrix:
      for pos in row:
        if pos == Game.WORKER or pos == Game.DOCKED_WORKER:
          return (x, y, pos)
        else:
          x = x + 1
      y = y + 1
      x = 0
    return None

  def can_move(self, x, y):
    worker_pos = self.worker()
    if not worker_pos:
      return False
    else:
      future_pos = self.get_content(worker_pos[0] + x, worker_pos[1] + y)
      return (future_pos not in [Game.WALL, Game.DOCKED_BOX, Game.BOX])

  def next(self, x, y):
    return self.get_content(self.worker()[0] + x, self.worker()[1] + y)

  def can_push(self, x, y):
    worker_pos = self.worker()
    if not worker_pos:
      return False
    else:
      return (self.next(x, y) in [Game.DOCKED_BOX, Game.BOX] and
              self.next(x + x, y + y) in [Game.FLOOR, Game.DOCK])

  def is_completed(self):
    for row in self.matrix:
      for cell in row:
        if cell == Game.BOX:
          return False
    return True

  def move_box(self, x, y, a, b):
    #        (x,y) -> move to do
    #        (a,b) -> box to move
    current_box = self.get_content(x, y)
    future_box = self.get_content(x + a, y + b)
    if current_box == Game.BOX and future_box == Game.FLOOR:
      self.set_content(x + a, y + b, Game.BOX)
      self.set_content(x, y, Game.FLOOR)
    elif current_box == Game.BOX and future_box == Game.DOCK:
      self.set_content(x + a, y + b, Game.DOCKED_BOX)
      self.set_content(x, y, Game.FLOOR)
    elif current_box == Game.DOCKED_BOX and future_box == Game.FLOOR:
      self.set_content(x + a, y + b, Game.BOX)
      self.set_content(x, y, Game.DOCK)
    elif current_box == Game.DOCKED_BOX and future_box == Game.DOCK:
      self.set_content(x + a, y + b, Game.DOCKED_BOX)
      self.set_content(x, y, Game.DOCK)

  def move(self, x, y):
    if self.can_move(x, y):
      current = self.worker()
      future = self.next(x, y)
      if current[2] == Game.WORKER and future == Game.FLOOR:
        self.set_content(current[0] + x, current[1] + y, Game.WORKER)
        self.set_content(current[0], current[1], Game.FLOOR)
      elif current[2] == Game.WORKER and future == Game.DOCK:
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
        self.set_content(current[0], current[1], Game.FLOOR)
      elif current[2] == Game.DOCKED_WORKER and future == Game.FLOOR:
        self.set_content(current[0] + x, current[1] + y, Game.WORKER)
        self.set_content(current[0], current[1], Game.DOCK)
      elif current[2] == Game.DOCKED_WORKER and future == Game.DOCK:
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
        self.set_content(current[0], current[1], Game.DOCK)
    elif self.can_push(x, y):
      current = self.worker()
      future = self.next(x, y)
      future_box = self.next(x + x, y + y)
      if (current[2] == Game.WORKER and future == Game.BOX and
          future_box == Game.FLOOR):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.FLOOR)
        self.set_content(current[0] + x, current[1] + y, Game.WORKER)
      elif (current[2] == Game.WORKER and future == Game.BOX and
            future_box == Game.DOCK):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.FLOOR)
        self.set_content(current[0] + x, current[1] + y, Game.WORKER)
      elif (current[2] == Game.WORKER and future == Game.DOCKED_BOX and
            future_box == Game.FLOOR):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.FLOOR)
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
      elif (current[2] == Game.WORKER and future == Game.DOCKED_BOX and
            future_box == Game.DOCK):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.FLOOR)
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
      if (current[2] == Game.DOCKED_WORKER and future == Game.BOX and
          future_box == Game.FLOOR):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.DOCK)
        self.set_content(current[0] + x, current[1] + y, Game.WORKER)
      elif (current[2] == Game.DOCKED_WORKER and future == Game.BOX and
            future_box == Game.DOCK):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.DOCK)
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
      elif (current[2] == Game.DOCKED_WORKER and future == Game.DOCKED_BOX and
            future_box == Game.FLOOR):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.DOCK)
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)
      elif (current[2] == Game.DOCKED_WORKER and future == Game.DOCKED_BOX and
            future_box == Game.DOCK):
        self.move_box(current[0] + x, current[1] + y, x, y)
        self.set_content(current[0], current[1], Game.DOCK)
        self.set_content(current[0] + x, current[1] + y, Game.DOCKED_WORKER)


class GameGui:
  BLOCK_PIXEL_SIZE = 32

  def __init__(self, levels_file, level_name):
    self.wall = pygame.image.load('images/wall.png')
    self.floor = pygame.image.load('images/floor.png')
    self.box = pygame.image.load('images/box.png')
    self.box_docked = pygame.image.load('images/box_docked.png')
    self.worker = pygame.image.load('images/worker.png')
    self.worker_docked = pygame.image.load('images/worker_dock.png')
    self.docker = pygame.image.load('images/dock.png')
    self.background = 255, 226, 191
    pygame.init()
    pygame.display.set_caption('Sokoban')
    start = pygame.display.set_mode((320, 240))
    level = 1
    self.game = Game(levels_file, level_name)
    self.size = self.game.load_size(GameGui.BLOCK_PIXEL_SIZE)
    self.surface = pygame.display.set_mode(self.size)
    self.sample_dim = 64
    self.render_fovea = True
    self.focal_pos = (self.size[1] // 2, self.size[0] // 2)
    self.fovea = np.zeros((self.sample_dim, self.sample_dim, 3), np.uint8)
    self.session = None
    self.praxis = Praxis()
    self.render_game()

  def Run(self):
    while True:
      events = pygame.event.get()
      if not events:
        time.sleep(0.050)
        continue

      for event in events:
        if event.type == pygame.QUIT: sys.exit(0)
        elif event.type == pygame.KEYDOWN:
          if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                           pygame.K_RIGHT):
            move = None
            if event.key == pygame.K_UP: move = (0, -1)
            elif event.key == pygame.K_DOWN: move = (0, 1)
            elif event.key == pygame.K_LEFT: move = (-1, 0)
            elif event.key == pygame.K_RIGHT: move = (1, 0)
            self.game.move(*move)
            if self.session:
              self.render_game()
              self.capture_session_event(move=move)
          elif event.key == pygame.K_q:
            sys.exit(0)
          elif event.key == pygame.K_k:
            self.screenshot()
          elif event.key == pygame.K_f:
            self.foveashot()
          elif event.key == pygame.K_r:
            self.render_fovea = not self.render_fovea
          elif event.key == pygame.K_s and not self.session:
            self.session = Session()
            self.render_game()
            self.capture_session_event()
          elif event.key == pygame.K_c and self.session:
            self.praxis.sessions.append(self.session)
            print(f'Captured session events: {len(self.session.events)}')
            self.session = None
          elif event.key == pygame.K_j and self.session:
            print('Rejected session')
            self.session = None
        elif event.type == pygame.MOUSEBUTTONDOWN:
          x, y = pygame.mouse.get_pos()
          old_focal_pos = self.focal_pos
          self.focal_pos = (y, x)
          if self.session:
            offset = ((self.focal_pos[0] - old_focal_pos[0]),
                      (self.focal_pos[1] - old_focal_pos[1]))
            self.render_game()
            self.capture_session_event(offset=offset)
      if not self.session:
        self.render_game()

  def get_fovea_state(self):
    return np.array(
        Image.fromarray(self.fovea).convert('LA').resize((16, 16),
                                                         Image.BICUBIC))

  def capture_session_event(self, move=(0, 0), offset=(0, 0)):
    assert self.session
    self.session.events.append(SessionEvent(self.fovea, move, offset))


  def foveashot(self):
    img = Image.fromarray(self.fovea).convert('LA')
    img = img.resize((16, 16), Image.BICUBIC)
    img.save('fovea.png')

  def screenshot(self):
    rgb = pygame.surfarray.array3d(self.surface)
    rgb = rgb.swapaxes(0, 1)
    Image.fromarray(rgb).save('screenshot.png')

  def extract_fovea(self):
    rgb = pygame.surfarray.array3d(self.surface)
    rgb = rgb.swapaxes(0, 1)

    substrate = np.zeros((rgb.shape[0] + self.sample_dim,
                          rgb.shape[1] + self.sample_dim, rgb.shape[2]),
                         rgb.dtype)
    half_sample_dim = self.sample_dim // 2
    substrate[half_sample_dim:half_sample_dim + rgb.shape[0], half_sample_dim:
              half_sample_dim + rgb.shape[1], :] = rgb

    start_row = self.focal_pos[0]
    start_col = self.focal_pos[1]
    end_row = start_row + self.sample_dim
    end_col = start_col + self.sample_dim
    self.fovea = substrate[start_row:end_row, start_col:end_col, :]
    assert self.fovea.shape == (self.sample_dim, self.sample_dim, 3)

  def render_game(self):
    self.surface.fill(self.background)
    x = 0
    y = 0
    for row in self.game.get_matrix():
      for char in row:
        if char == Game.FLOOR:  #floor
          self.surface.blit(self.floor, (x, y))
        elif char == Game.WALL:  #wall
          self.surface.blit(self.wall, (x, y))
        elif char == Game.WORKER:  #worker on floor
          self.surface.blit(self.worker, (x, y))
        elif char == Game.DOCK:  #dock
          self.surface.blit(self.docker, (x, y))
        elif char == Game.DOCKED_BOX:  #box on dock
          self.surface.blit(self.box_docked, (x, y))
        elif char == Game.BOX:  #box
          self.surface.blit(self.box, (x, y))
        elif char == Game.DOCKED_WORKER:  #worker on dock
          self.surface.blit(self.worker_docked, (x, y))
        x = x + GameGui.BLOCK_PIXEL_SIZE
      x = 0
      y = y + GameGui.BLOCK_PIXEL_SIZE

    if self.render_fovea:
      render_dim = self.sample_dim + 2
      x_start = self.focal_pos[1] - render_dim / 2
      y_start = self.focal_pos[0] - render_dim / 2
      border_width = 1
      fovea_color = (0, 0, 255)
      if self.session:
        fovea_color = (255, 0, 0)
      pygame.draw.rect(self.surface, fovea_color,
                       (x_start, y_start, render_dim, render_dim), border_width)
    pygame.display.update()
    self.extract_fovea()


def main(argv):

  def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)

  signal.signal(signal.SIGINT, handle_pdb)
  gui = GameGui('sandbox.ascii', 'Arena')
  gui.Run()


if __name__ == '__main__':
  app.run(main)
