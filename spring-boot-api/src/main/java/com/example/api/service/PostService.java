package com.example.api.service;

import com.example.api.model.Post;
import com.example.api.repository.PostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PostService {
    
    @Autowired
    private PostRepository postRepository;
    
    public List<Post> getAllPosts() {
        return postRepository.findAll();
    }
    
    public Optional<Post> getPostById(Long id) {
        return postRepository.findById(id);
    }
    
    public List<Post> getPostsByAuthorId(Long authorId) {
        return postRepository.findByAuthorId(authorId);
    }
    
    public List<Post> getPublishedPosts() {
        return postRepository.findByPublished(true);
    }
    
    public Post createPost(Post post) {
        return postRepository.save(post);
    }
    
    public Post updatePost(Long id, Post postDetails) {
        Post post = postRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Post not found with id: " + id));
        
        post.setTitle(postDetails.getTitle());
        post.setContent(postDetails.getContent());
        post.setCategory(postDetails.getCategory());
        post.setPublished(postDetails.getPublished());
        
        return postRepository.save(post);
    }
    
    public void deletePost(Long id) {
        postRepository.deleteById(id);
    }
    
    public Post incrementViewCount(Long id) {
        Post post = postRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Post not found with id: " + id));
        post.setViewCount(post.getViewCount() + 1);
        return postRepository.save(post);
    }
}
