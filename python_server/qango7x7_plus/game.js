num_squares = 37
const body_rect = document.body.getBoundingClientRect();
const board_size = Math.min(body_rect.height * 0.85, body_rect.width * 0.5);
const board = document.getElementById("board");
board.style.width = board_size + "px";
board.style.height = board_size + "px";
board.style.top = (body_rect.height/2 - board_size/2)+"px";
const tbody = board.children[0];
for (var tr of tbody.children){
    tr.style.height = (board_size/Math.sqrt(num_squares)) + "px";
    for (var td of tr.children){
        td.style.width = (board_size/Math.sqrt(num_squares)) + "px";
        td.style.height = "100%";
    }
}
function move_int_to_str(int_move){
    removals = [0,0,3,3,3,8,29,34,34,34]
    for (var i = 0; i<removals.length; i++){
        if (removals[i]>int_move){
            break
        }
    }
    console.log(int_move,i)
    int_move+=i
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[int_move % 7]+(parseInt(int_move/7)+1)
}

function get_squares(){
    var tbody = document.getElementById("board").children[0]
    var tds = []
    for (var i = 0; i < tbody.children.length; i++){
        var tr = tbody.children[i]
        for (var j = 0; j < tr.children.length; j++){
            if (tr.children[j].className!="invis"){
                tds.push(tr.children[j])
            }
        }
    }
    return tds
}