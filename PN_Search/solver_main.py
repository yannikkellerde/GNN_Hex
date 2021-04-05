import os,sys,json
from flask import render_template,escape
base_path = os.path.abspath(os.path.dirname(__file__))

class Solver_handler():
    def __init__(self):
        self.no_json_games = ("qango6x6","qango7x7","qango7x7_plus")
        self.json_path = os.path.join(base_path,"json_games")
    def get_table_entry(self,game_name):
        game_name = escape(game_name)
        return f"""
        <tr>
            <td>{game_name}</td>
            <td><a href="/solver/creator/{game_name}">Edit</a></td>
            <td><a href="/solver/analyze/{game_name}">Analyze</a></td>
            <td><a href="#" onclick='del_game("{game_name}")'>Delete</a></td>
        <tr/>
        """
    def do_GET(self):
        table = ""
        for game in os.listdir(self.json_path):
            table += self.get_table_entry(".".join(game.split(".")[:-1]))
        return render_template("solver_main.html",table_body=table)
    def do_POST(self,data):
        if "delete" in data:
            if (os.path.isfile(os.path.join(self.json_path,data["delete"]+".json"))):
                os.remove(os.path.join(self.json_path,data["delete"]+".json"))
        return json.dumps({})

solver_handler = Solver_handler()