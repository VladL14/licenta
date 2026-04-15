import { Component, ChangeDetectorRef, NgZone} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Emotion } from './emotion';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  userText: string = '';
  imageCode: string  = '';
  currentAvatar: string = '';
  messages: {role: string, content: string}[] = [];
  emotionResult: any = null;
  showExplanation: boolean = false;
  isLoading: boolean = false;

  constructor(private emotionService: Emotion, private cdRef: ChangeDetectorRef, private ngZone: NgZone) {}

  imageToBase64(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.ngZone.run(() => {
          this.imageCode = e.target.result;
          this.currentAvatar = e.target.result;
          this.cdRef.detectChanges();
        });
      };
      reader.readAsDataURL(file);
    }
  }

  sendMessage(){
    if(this.userText.trim() === '' || !this.imageCode) return;
    const text = this.userText;
    this.messages.push({role: 'user', content: text});
    this.userText = '';
    this.isLoading = true;
    this.showExplanation = false;
    this.cdRef.detectChanges();
    const history = this.messages.slice(0, -1);
    this.emotionService.sendMessage(text, this.imageCode, history).subscribe({
      next: (response) => {
        this.ngZone.run(() => {
          this.emotionResult = response;
          this.currentAvatar = response.generated_image || this.imageCode;
          this.messages.push({role: 'bot', content: response.bot_response});
          this.isLoading = false;
          this.cdRef.detectChanges();
        });
      },error: (error) => {
        this.ngZone.run(() => {
          console.error('Error:', error);
          this.isLoading = false;
          this.cdRef.detectChanges();
        });
      }
  });
  }
  toggleExplanation() {
    this.showExplanation = !this.showExplanation;
  }
}

