import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class Emotion {
  private apiUrl = 'http://127.0.0.1:8000/chat';
  constructor(private http: HttpClient) {}

  sendMessage(userText: string, ImageCode: string, chatHistory: any[]): Observable<any> {
    const body = {
      text: userText,
      image_code: ImageCode,
      history: chatHistory
    };
    return this.http.post(this.apiUrl, body);
  }
}