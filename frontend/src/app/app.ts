import { Component, ChangeDetectorRef, NgZone, ViewChild, ElementRef, AfterViewChecked, AfterViewInit, HostListener, OnDestroy, Inject, PLATFORM_ID } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Emotion } from './emotion';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewChecked, AfterViewInit, OnDestroy {
  @ViewChild('chatScroll') private chatScrollContainer!: ElementRef;
  @ViewChild('networkCanvas') private canvasRef!: ElementRef<HTMLCanvasElement>;
  
  userText: string = '';
  imageCode: string  = '';
  currentAvatar: string = '';
  messages: {role: string, content: string}[] = [];
  emotionResult: any = null;
  showExplanation: boolean = false;
  isLoading: boolean = false;
  isAvatarUploaded: boolean = false;

  private ctx!: CanvasRenderingContext2D;
  private particles: Particle[] = [];
  private animationFrameId: number = 0;
  private isBrowser: boolean;

  constructor(
    private emotionService: Emotion, 
    private cdRef: ChangeDetectorRef, 
    private ngZone: NgZone,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {
    this.isBrowser = isPlatformBrowser(this.platformId);
  }

  ngAfterViewInit() {
    if (this.isBrowser) {
      this.initCanvas();
    }
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  ngOnDestroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  @HostListener('window:resize')
  onResize() {
    if (this.isBrowser) {
      this.resizeCanvas();
    }
  }

  private initCanvas() {
    const canvas = this.canvasRef.nativeElement;
    const context = canvas.getContext('2d');
    if (!context) return;
    this.ctx = context;
    this.resizeCanvas();
    
    // Initialize particles
    const particleCount = Math.floor(window.innerWidth / 10);
    this.particles = [];
    for (let i = 0; i < particleCount; i++) {
      this.particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5
      });
    }

    this.ngZone.runOutsideAngular(() => {
      this.animate();
    });
  }

  private resizeCanvas() {
    const canvas = this.canvasRef.nativeElement;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  private animate = () => {
    if (!this.isBrowser || !this.canvasRef) return;
    const canvas = this.canvasRef.nativeElement;
    if (!canvas) return;

    this.ctx.clearRect(0, 0, canvas.width, canvas.height);

    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    this.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      this.ctx.fill();
    });

    this.ctx.lineWidth = 0.5;
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const dx = this.particles[i].x - this.particles[j].x;
        const dy = this.particles[i].y - this.particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 120) {
          this.ctx.strokeStyle = `rgba(255, 255, 255, ${1 - dist / 120})`;
          this.ctx.beginPath();
          this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
          this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
          this.ctx.stroke();
        }
      }
    }

    this.animationFrameId = requestAnimationFrame(this.animate);
  };

  scrollToBottom(): void {
    try {
      if (this.chatScrollContainer) {
        this.chatScrollContainer.nativeElement.scrollTop = this.chatScrollContainer.nativeElement.scrollHeight;
      }
    } catch(err) { }
  }

  imageToBase64(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.ngZone.run(() => {
          this.imageCode = e.target.result;
          this.currentAvatar = e.target.result;
          this.isAvatarUploaded = true;
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
  
  resetAvatar() {
    this.isAvatarUploaded = false;
    this.imageCode = '';
    this.currentAvatar = '';
    this.messages = [];
    this.emotionResult = null;
  }
}
