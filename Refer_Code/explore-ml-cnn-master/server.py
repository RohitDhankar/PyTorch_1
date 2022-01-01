import http.server
import io
import json
from cnn import start_training, test

class ServerHandler(http.server.BaseHTTPRequestHandler):
  def do_POST(self):
    response_code = 200
    response = ""
    var_len = int(self.headers.get('Content-Length'))
    content = self.rfile.read(var_len)

    payload = json.loads(content)
    print(payload)

    if 'train' in self.path:
      print('train')
      start_training()
    elif 'test' in self.path:
      print('test')
      response = test(payload['image'], payload['label'])
    else:
      response_code = 400
    
    b = io.BytesIO(json.dumps(response).encode('utf-8'))
    view = b.getbuffer()
    
    self.send_response(response_code)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(view)

try:
  server_address = ('', 8000)
  server = http.server.HTTPServer(server_address, ServerHandler)
  server.serve_forever()
except KeyboardInterrupt:
  server.socket.close()
