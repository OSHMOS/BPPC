package com.example.api.controller;

import com.example.api.model.Post;
import com.example.api.service.PostService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/posts")
public class PostController {
    
    @Autowired
    private PostService postService;
    
    @GetMapping
    public ResponseEntity<List<Post>> getAllPosts() {
        return ResponseEntity.ok(postService.getAllPosts());
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<Post> getPostById(@PathVariable Long id) {
        return postService.getPostById(id)
                .map(post -> {
                    postService.incrementViewCount(id);
                    return ResponseEntity.ok(post);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/author/{authorId}")
    public ResponseEntity<List<Post>> getPostsByAuthorId(@PathVariable Long authorId) {
        return ResponseEntity.ok(postService.getPostsByAuthorId(authorId));
    }
    
    @GetMapping("/published")
    public ResponseEntity<List<Post>> getPublishedPosts() {
        return ResponseEntity.ok(postService.getPublishedPosts());
    }
    
    @PostMapping
    public ResponseEntity<Post> createPost(@Valid @RequestBody Post post) {
        Post createdPost = postService.createPost(post);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdPost);
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<Post> updatePost(@PathVariable Long id, @Valid @RequestBody Post post) {
        try {
            Post updatedPost = postService.updatePost(id, post);
            return ResponseEntity.ok(updatedPost);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deletePost(@PathVariable Long id) {
        postService.deletePost(id);
        return ResponseEntity.noContent().build();
    }
}
